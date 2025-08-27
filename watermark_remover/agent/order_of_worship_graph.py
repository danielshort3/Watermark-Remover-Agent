"""LangGraph pipeline for processing an order of worship PDF.

This graph extends the original Watermark Remover pipeline to handle
"order of worship" documents.  Given a natural‑language instruction
containing the name of a PDF file and a target instrument, the graph
extracts a list of songs from the PDF, annotates each song with the
desired instrument (respecting any per‑song overrides specified in the
instruction) and then runs the scraping, watermark removal, upscaling
and PDF assembly tools sequentially for each song.  The result is a
list of individual PDF files, one per song, saved to disk.

Example input::

    {
        "user_input": "Use the order of worship titled 'August 24, 2025.pdf' and extract songs for French Horn, then scrape each of them, remove the watermarks, upscale them, and assemble each song into an individual PDF."
    }

The parser node will identify the PDF filename (``August 24, 2025.pdf``),
the default instrument (``French Horn``) and any per‑song overrides
(e.g. ``make sure the second song is for Oboe instead``).  The
extractor node reads the specified PDF, parses the song list and
attaches the instrument information.  The processor node then loops
over the songs, invoking the same chain of tools used by the
Watermark Remover pipeline for each title.
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph, START, END

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)

# ---------------------------------------------------------------------------
# PDF parsing helpers (adapted from pdf_parse.py)
#
# We copy the PDF extraction logic here to avoid adding extra runtime
# dependencies.  These functions attempt to extract text from a PDF via
# multiple libraries and then parse lines of the form:
#     "Holy Water [ We the Kingdom in F ]"
# The resulting dictionary maps indices to song metadata (title, artist,
# key).  The ``instrument`` field will be attached later.
# ---------------------------------------------------------------------------

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore

# Import the Ollama-backed LLM runner to parse natural-language instructions.
# Falls back gracefully if Ollama or the agent wrapper are unavailable.
try:
    from watermark_remover.agent.graph_ollama import run_instruction as _ollama_parse
except Exception:
    _ollama_parse = None  # LLM parsing disabled; will fall back to regex

def _read_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF using one of several fallback libraries.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file on disk.

    Returns
    -------
    str
        The concatenated text extracted from all pages.

    Raises
    ------
    RuntimeError
        If none of the supported PDF libraries are available.
    """
    # Use pdfplumber if installed
    if pdfplumber is not None:
        try:
            text_chunks: List[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_chunks.append(page.extract_text() or "")
            return "\n".join(text_chunks)
        except Exception:
            pass
    # Fallback to PyPDF2
    if PdfReader is not None:
        try:
            reader = PdfReader(pdf_path)  # type: ignore[call-arg]
            text_chunks: List[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                text_chunks.append(txt)
            return "\n".join(text_chunks)
        except Exception:
            pass
    # Fallback to pdfminer.six
    if pdfminer_extract_text is not None:
        try:
            txt = pdfminer_extract_text(pdf_path)  # type: ignore[call-arg]
            return txt or ""
        except Exception:
            pass
    raise RuntimeError(
        "Could not read PDF text. Please install pdfplumber, PyPDF2, or pdfminer.six."
    )


_SONG_LINE_RE = re.compile(
    r"""^\s*
        (?:\d{1,2}:\d{2}\s+)?                # optional leading duration
        (?P<title>.+?)                       # song title (non-greedy)
        \s*\[\s*
        (?P<artist>.+?)                      # artist/arranger text up to ' in '
        \s+in\s+
        (?P<key>[A-Ga-g](?:[#b])?(?:\s*(?:maj(?:or)?|min(?:or)?|m))?)  # musical key
        \s*\]\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_key(key: str) -> str:
    """Normalise a musical key to a compact form.

    Examples
    --------
    >>> _normalize_key('Bb major')
    'Bb'
    >>> _normalize_key('C# minor')
    'C#m'
    >>> _normalize_key('G m')
    'Gm'
    """
    s = key.strip().replace("♭", "b").replace("♯", "#")
    base_match = re.match(r"^[A-Ga-g](?:[#b])?", s)
    base = (base_match.group(0).upper() if base_match else s.strip().upper())
    is_minor = bool(re.search(r"(?:\bmin(?:or)?\b|\bm\b)$", s, re.IGNORECASE))
    return base + ("m" if is_minor else "")


def extract_songs_from_text(text: str) -> Dict[int, Dict[str, str]]:
    """Parse song lines from raw PDF text into a dict keyed by index."""
    songs: List[Dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Only attempt to parse lines containing brackets
        if "[" not in line or "]" not in line:
            continue
        m = _SONG_LINE_RE.match(line)
        if not m:
            continue
        title = m.group("title").strip()
        artist = m.group("artist").strip()
        key = _normalize_key(m.group("key"))
        songs.append({"title": title, "artist": artist, "key": key})
    return {i: song for i, song in enumerate(songs)}


def extract_songs_from_pdf(pdf_path: str) -> Dict[int, Dict[str, str]]:
    """Extract songs from the given PDF file.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file on disk.

    Returns
    -------
    Dict[int, Dict[str, str]]
        A mapping from song index to a dictionary containing ``title``,
        ``artist`` and ``key``.
    """
    text = _read_pdf_text(pdf_path)
    return extract_songs_from_text(text)


# ---------------------------------------------------------------------------
# Natural language parsing helpers
# ---------------------------------------------------------------------------

_ORDINAL_MAP = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
}


def _parse_user_input(instruction: str) -> Tuple[str, str, Dict[int, str]]:
    """Parse the user instruction for PDF filename, default instrument and overrides.

    Parameters
    ----------
    instruction : str
        The raw user instruction.

    Returns
    -------
    Tuple[str, str, Dict[int, str]]
        A tuple containing:
        * pdf_name: the filename of the order of worship PDF (with extension)
        * default_instrument: the instrument to use for all songs unless overridden
        * overrides: a mapping from song index (0‑based) to an instrument

    Notes
    -----
    The parser uses simple heuristics to extract relevant information.
    It looks for a quoted filename ending with ``.pdf`` following the
    word ``titled``.  It then finds phrases like ``songs for X`` to
    determine the default instrument.  Per‑song overrides are detected
    using patterns such as ``second song ... for Oboe`` or
    ``make sure the 2nd song is for Oboe``.
    """
    pdf_name = ""
    default_instrument = ""
    overrides: Dict[int, str] = {}

    # Extract PDF name
    pdf_match = re.search(r"titled\s+['\"]([^'\"]+\.pdf)['\"]", instruction, re.IGNORECASE)
    if pdf_match:
        pdf_name = pdf_match.group(1).strip()

    # Extract default instrument (look for "songs for X" or "for X" after "songs")
    instr_match = re.search(
        r"(?:extract\s+songs\s+for|songs\s+for|for)\s+([^,\.]+?)\s*(?:,|\.|…| and |$)",
        instruction,
        re.IGNORECASE,
    )
    if instr_match:
        default_instrument = instr_match.group(1).strip()

    # Extract per‑song overrides
    # Pattern matches e.g. "second song is for Oboe", "2nd song is for Oboe"
    override_pattern = re.compile(
        r"(?:(?:the|that)\s+)?((?:\d+)(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+song[^,\.]*?for\s+([A-Za-z ]+?)\b",
        re.IGNORECASE,
    )
    for om in override_pattern.finditer(instruction):
        ordinal_raw = om.group(1).lower()
        instr = om.group(2).strip()
        idx: int = -1
        # Convert ordinal word to index
        if ordinal_raw.isdigit():
            try:
                idx = int(re.sub(r"(?:st|nd|rd|th)$", "", ordinal_raw)) - 1
            except Exception:
                idx = -1
        else:
            idx = _ORDINAL_MAP.get(re.sub(r"(?:st|nd|rd|th)$", "", ordinal_raw), -1)
        if idx >= 0:
            overrides[idx] = instr
    return pdf_name, default_instrument, overrides


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from the user instruction into the state.

    Prefer an Ollama-backed LLM to parse the natural-language instruction into
    structured fields (pdf_name, default_instrument, overrides). If the LLM is
    unavailable or returns invalid JSON, fall back to the regex-based parser.
    """
    new_state: Dict[str, Any] = dict(state)
    instruction = (new_state.get("user_input") or "").strip()

    # Only parse if keys are missing
    if new_state.get("pdf_name") and new_state.get("default_instrument"):
        # Still merge any overrides if present in the instruction
        _, _, overrides_fallback = _parse_user_input(instruction)
        merged = dict(new_state.get("overrides", {}) or {})
        merged.update(overrides_fallback or {})
        new_state["overrides"] = merged
        return new_state

    pdf_name = ""
    default_instrument = ""
    overrides: Dict[int, str] = {}

    # --- LLM-first parse ---
    if _ollama_parse and instruction:
        try:
            llm_prompt = (
                "You are given a user instruction describing how to process an 'order of worship' PDF. "
                "Extract:\n"
                "  - pdf_name: the exact PDF filename including the .pdf extension\n"
                "  - default_instrument: the instrument to use for all songs unless overridden\n"
                "  - overrides: a mapping of zero-based song indices to an instrument name, for any exceptions\n"
                "Return ONLY valid JSON with keys exactly: pdf_name, default_instrument, overrides.\n\n"
                f"Instruction: {instruction}"
            )
            llm_output = _ollama_parse(llm_prompt)

            # Extract JSON from the model output (string or dict)
            data = None
            if isinstance(llm_output, dict):
                data = llm_output
            else:
                s = str(llm_output)
                # Try to capture the outermost JSON block
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(s[start : end + 1])
                else:
                    # As a last resort try parsing the whole string
                    data = json.loads(s)

            if isinstance(data, dict):
                pdf_name = (data.get("pdf_name") or "").strip()
                default_instrument = (data.get("default_instrument") or "").strip()
                raw_overrides = (data.get("overrides") or {}) if isinstance(data.get("overrides"), dict) else {}

                # Coerce override keys to zero-based indices (handle "2", "2nd", "second", etc.)
                for k, v in raw_overrides.items():
                    idx = None
                    # numeric key e.g. "2" -> index 2
                    if isinstance(k, int):
                        idx = k
                    else:
                        ks = str(k).strip().lower()
                        # remove trailing ordinal suffix
                        ks = re.sub(r"(?:st|nd|rd|th)$", "", ks)
                        # remove 'song' word or leading 'the'
                        ks = re.sub(r"^(?:the\s+)?", "", ks)
                        ks = ks.replace("song", "").strip()
                        if ks.isdigit():
                            idx = int(ks)
                        else:
                            idx = _ORDINAL_MAP.get(ks, None)
                    if idx is not None:
                        overrides[int(idx)] = str(v).strip()
        except Exception:
            # swallow and fall back to regex-based parsing
            pdf_name = default_instrument = ""
            overrides = {}

    # --- Fallback: heuristic/regex parser ---
    if not pdf_name or not default_instrument:
        rx_pdf, rx_instr, rx_overrides = _parse_user_input(instruction)
        pdf_name = pdf_name or rx_pdf
        default_instrument = default_instrument or rx_instr
        overrides = {**overrides, **(rx_overrides or {})}

    # Update state (existing values keep precedence if already set)
    if pdf_name and not new_state.get("pdf_name"):
        new_state["pdf_name"] = pdf_name
    if default_instrument and not new_state.get("default_instrument"):
        new_state["default_instrument"] = default_instrument

    merged_overrides = dict(new_state.get("overrides", {}) or {})
    merged_overrides.update(overrides or {})
    new_state["overrides"] = merged_overrides

    return new_state


def extract_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Read the specified PDF and attach song metadata to the state.

    The node looks up ``pdf_name`` in the state and loads the PDF
    from disk.  It then parses the song list and attaches an
    ``instrument`` field to each entry.  The instrument is determined by
    the ``default_instrument`` and any per‑song ``overrides``.  The
    resulting list of songs is stored under the ``songs`` key in the
    state.
    """
    new_state: Dict[str, Any] = dict(state)
    pdf_name = new_state.get("pdf_name")
    default_instrument = new_state.get("default_instrument", "")
    overrides: Dict[int, str] = new_state.get("overrides", {}) or {}
    if not pdf_name:
        # Nothing to do if no PDF specified
        new_state["songs"] = {}
        return new_state
    # Construct an absolute path; if the file does not exist in the CWD,
    # attempt to locate it under a user‑specified directory (pdf_root or input_dir).
    # When running inside Docker the PDFs are typically mounted under /app/input.
    pdf_path = pdf_name
    if not os.path.isabs(pdf_path):
        root_dir = new_state.get("pdf_root") or new_state.get("input_dir") or "input"
        candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
        if os.path.exists(candidate):
            pdf_path = candidate
        else:
            pdf_path = os.path.join(os.getcwd(), pdf_path)
    songs: Dict[int, Dict[str, Any]] = {}
    try:
        parsed = extract_songs_from_pdf(pdf_path)
        for idx, meta in parsed.items():
            instrument = overrides.get(idx, default_instrument)
            songs[idx] = {
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "key": meta.get("key", ""),
                "instrument": instrument or "",
            }
    except Exception:
        # If parsing fails, log nothing
        songs = {}
    new_state["songs"] = songs
    return new_state


def process_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process each song sequentially using the scraping pipeline.

    For every song found in the ``songs`` dict, this node invokes the
    ``scrape_music``, ``remove_watermark``, ``upscale_images`` and
    ``assemble_pdf`` tools in order.  It collects the resulting PDF
    filenames into a list stored under ``final_pdfs``.  If any step
    fails for a particular song, the corresponding entry in the result
    list will be ``None``.
    """
    new_state: Dict[str, Any] = dict(state)
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    final_pdfs: List[str | None] = []
    input_dir_override = new_state.get("input_dir", "data/samples")
    for idx in sorted(songs.keys()):
        song = songs[idx]
        title = song.get("title", "Unknown Title")
        instrument = song.get("instrument", "Unknown Instrument")
        key = song.get("key", "Unknown Key")
        # Run through the tools
        try:
            download_dir = scrape_music.invoke(
                {
                    "title": title,
                    "instrument": instrument,
                    "key": key,
                    "input_dir": input_dir_override,
                }
            )
        except Exception:
            download_dir = None
        if not download_dir:
            final_pdfs.append(None)
            continue
        try:
            processed_dir = remove_watermark.invoke({"input_dir": download_dir})
        except Exception:
            processed_dir = None
        if not processed_dir:
            final_pdfs.append(None)
            continue
        try:
            upscaled_dir = upscale_images.invoke({"input_dir": processed_dir})
        except Exception:
            upscaled_dir = None
        if not upscaled_dir:
            final_pdfs.append(None)
            continue
        # Compose output filename; include instrument and key to avoid collisions
        output_name = f"{title}_{instrument}_{key}.pdf".replace(" ", "_")
        try:
            pdf_path = assemble_pdf.invoke(
                {
                    "image_dir": upscaled_dir,
                    "output_pdf": output_name,
                }
            )
        except Exception:
            pdf_path = None
        final_pdfs.append(
            pdf_path if pdf_path and isinstance(pdf_path, str) else None
        )
    new_state["final_pdfs"] = final_pdfs
    return new_state


def compile_graph() -> Any:
    """Construct and compile the order‑of‑worship graph.

    Returns
    -------
    Any
        A compiled graph ready for execution or visualisation with
        LangGraph.  The graph expects an initial state containing at
        least a ``user_input`` field.
    """
    graph = StateGraph(dict)
    graph.add_node("parser", parser_node)
    graph.add_node("extractor", extract_songs_node)
    graph.add_node("processor", process_songs_node)
    # Wire the nodes in sequence: parse -> extract -> process
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "extractor")
    graph.add_edge("extractor", "processor")
    graph.add_edge("processor", END)
    return graph.compile()


# Expose a compiled graph instance for LangGraph CLI discovery.
graph = compile_graph()
