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

# Import the global sanitiser from the tools module.  This helper converts
# free‑form names (titles, instruments, keys) into safe filesystem names.
from watermark_remover.agent.tools import sanitize_title, SCRAPE_METADATA, TEMP_DIRS

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)

# Attempt to import the Ollama-backed LLM runner.  If unavailable, we
# will fall back to heuristic parsing.  The run_instruction helper
# invokes a LangChain agent via Ollama to interpret natural‑language
# instructions into structured JSON.
try:
    from watermark_remover.agent.graph_ollama import run_instruction  # type: ignore
except Exception:
    run_instruction = None  # type: ignore

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

    This node reads the ``user_input`` key from the incoming state and
    populates ``pdf_name``, ``default_instrument``, ``overrides`` and
    ``order_folder`` keys.  If any of these fields already exist in the
    state they are left unchanged.  Parsing is handled solely by the
    Ollama‑powered LLM; there is no manual fallback.  If the LLM fails to
    produce values, the corresponding keys remain unset and the graph
    may fail downstream.
    """
    new_state: Dict[str, Any] = dict(state)
    instruction = new_state.get("user_input", "") or ""
    # Only parse if keys are missing
    if not new_state.get("pdf_name") or not new_state.get("default_instrument") or not new_state.get("order_folder"):
        # Attempt LLM parsing if available
        pdf_name = ""
        default_instrument = ""
        overrides: Dict[int, str] = {}
        order_folder = ""
        if run_instruction is not None and instruction:
            try:
                llm_prompt = (
                    "You are given a user instruction describing how to process an order of worship PDF. "
                    "Extract the name of the PDF file (with .pdf extension) as 'pdf_name', the default instrument as "
                    "'default_instrument', a mapping of zero-based song indices to instruments as 'overrides' (if any), "
                    "and compute 'order_folder' based on the pdf_name. The order_folder should be the date portion of "
                    "the pdf_name formatted as MM_DD_YYYY (e.g. 'August 24, 2025.pdf' -> '08_24_2025'). "
                    "If the pdf_name does not contain a date, you may return a reasonable folder name by replacing "
                    "non-alphanumeric characters with underscores and trimming. Return ONLY a valid JSON object with keys: "
                    "pdf_name, default_instrument, overrides, order_folder."
                    "\nInstruction: " + instruction
                )
                llm_output = run_instruction(llm_prompt)
                data = None
                if isinstance(llm_output, dict):
                    data = llm_output
                else:
                    s = str(llm_output)
                    start = s.find("{")
                    end = s.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        import json as _json
                        data = _json.loads(s[start : end + 1])
                    else:
                        import json as _json
                        data = _json.loads(s)
                if isinstance(data, dict):
                    pdf_name = (data.get("pdf_name") or "").strip()
                    default_instrument = (data.get("default_instrument") or "").strip()
                    order_folder = (data.get("order_folder") or "").strip()
                    raw_overrides = data.get("overrides") or {}
                    if isinstance(raw_overrides, dict):
                        for k, v in raw_overrides.items():
                            idx = None
                            # normalise key indexes: handle numeric or ordinal strings
                            if isinstance(k, int):
                                idx = k
                            else:
                                ks = str(k).strip().lower()
                                ks = re.sub(r"(?:st|nd|rd|th)$", "", ks)
                                ks = ks.replace("song", "").strip()
                                if ks.isdigit():
                                    idx = int(ks)
                                else:
                                    idx = _ORDINAL_MAP.get(ks, None)
                            if idx is not None:
                                overrides[int(idx)] = str(v).strip()
            except Exception:
                # ignore LLM parsing errors
                pdf_name = default_instrument = order_folder = ""
                overrides = {}
        # Do not attempt to parse the instruction heuristically.  If the LLM
        # fails to return the necessary fields, leave them unset rather than
        # inferring values via regex.  This avoids inconsistent behaviour
        # between manual parsing and LLM parsing.  The graph will either
        # succeed with the LLM output or halt due to missing metadata.
        # Update state with parsed values if not already present
        if pdf_name and not new_state.get("pdf_name"):
            new_state["pdf_name"] = pdf_name
        if default_instrument and not new_state.get("default_instrument"):
            new_state["default_instrument"] = default_instrument
        if order_folder:
            # Sanitize order_folder to a safe directory name using the global sanitiser
            try:
                safe_order = sanitize_title(order_folder)
            except Exception:
                safe_order = order_folder
            new_state["order_folder"] = safe_order
        # Merge overrides
        current_overrides: Dict[int, str] = dict(new_state.get("overrides", {}))
        current_overrides.update(overrides)
        new_state["overrides"] = current_overrides
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
    # Construct absolute path; if the file does not exist in the CWD,
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


def scrape_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape sheet music for all songs before further processing.

    This node iterates over every song in the ``songs`` dict and calls
    ``scrape_music``.  The resulting download directory (or ``None`` on
    failure) is stored under ``download_dir`` for each song.  No other
    processing (watermark removal, upscaling or PDF assembly) occurs at
    this stage.  Temporary directories recorded during scraping are not
    cleaned up here; they will be removed in later stages.
    """
    new_state: Dict[str, Any] = dict(state)
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    input_dir_override = new_state.get("input_dir", "data/samples")
    for idx, song in songs.items():
        title = song.get("title", "Unknown Title")
        instrument = song.get("instrument", "Unknown Instrument")
        key = song.get("key", "Unknown Key")
        download_dir = None
        try:
            download_dir = scrape_music.invoke({
                "title": title,
                "instrument": instrument,
                "key": key,
                "input_dir": input_dir_override,
            })
        except Exception:
            download_dir = None
        song["download_dir"] = download_dir
    new_state["songs"] = songs
    return new_state


def remove_watermarks_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove watermarks for all previously scraped songs.

    This node reads the ``download_dir`` for each song (populated by
    ``scrape_songs_node``) and invokes ``remove_watermark`` on that
    directory.  The resulting directory containing watermark‑free images
    is stored under ``watermark_dir`` for each song.  Songs without a
    valid ``download_dir`` propagate ``None``.
    """
    new_state: Dict[str, Any] = dict(state)
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    for idx, song in songs.items():
        download_dir = song.get("download_dir")
        processed_dir = None
        if download_dir:
            try:
                processed_dir = remove_watermark.invoke({"input_dir": download_dir})
            except Exception:
                processed_dir = None
        song["watermark_dir"] = processed_dir
    new_state["songs"] = songs
    return new_state


def upscale_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Upscale images for all songs after watermark removal.

    This node iterates over ``watermark_dir`` for each song (set by
    ``remove_watermarks_node``) and calls ``upscale_images``.  The
    resulting upscaled directory is stored under ``upscaled_dir``.  If
    watermark removal failed, the ``upscaled_dir`` will be ``None``.
    """
    new_state: Dict[str, Any] = dict(state)
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    for idx, song in songs.items():
        processed_dir = song.get("watermark_dir")
        upscaled_dir = None
        if processed_dir:
            try:
                upscaled_dir = upscale_images.invoke({"input_dir": processed_dir})
            except Exception:
                upscaled_dir = None
        song["upscaled_dir"] = upscaled_dir
    new_state["songs"] = songs
    return new_state


def assemble_pdfs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble final PDFs for all songs after upscaling.

    This node uses the ``upscaled_dir`` for each song to call
    ``assemble_pdf``.  It builds consistent filenames using the actual
    key and instrument selected during scraping (from ``SCRAPE_METADATA``)
    and writes each PDF into both the ``output/music`` hierarchy and the
    ``output/orders/<date_folder>`` directory.  Temporary directories
    recorded during scraping are removed at the end of this node.
    The list of final PDF paths is stored under ``final_pdfs``.
    """
    new_state: Dict[str, Any] = dict(state)
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    final_pdfs: List[str | None] = []
    # Determine the order folder.  Use LLM‑derived order_folder if present
    date_folder = new_state.get("order_folder") or "unknown_date"
    if date_folder == "unknown_date":
        pdf_name: str = new_state.get("pdf_name", "") or ""
        if pdf_name:
            base = os.path.splitext(os.path.basename(pdf_name))[0]
            try:
                import datetime
                dt = datetime.datetime.strptime(base, "%B %d, %Y")
                date_folder = dt.strftime("%m_%d_%Y")
            except Exception:
                cleaned = re.sub(r"[^0-9]+", "_", base)
                cleaned = re.sub(r"_+", "_", cleaned).strip("_")
                if cleaned:
                    date_folder = cleaned
    for idx in sorted(songs.keys()):
        song = songs[idx]
        upscaled_dir = song.get("upscaled_dir")
        title = song.get("title", "Unknown Title")
        instrument = song.get("instrument", "Unknown Instrument")
        key = song.get("key", "Unknown Key")
        pdf_path = None
        if upscaled_dir:
            # Derive actual key/instrument from metadata if available
            actual_key = SCRAPE_METADATA.get("key") or key
            actual_instrument = SCRAPE_METADATA.get("instrument") or instrument
            # Persist actual values back into metadata for assemble_pdf
            try:
                SCRAPE_METADATA['key'] = actual_key
                SCRAPE_METADATA['instrument'] = actual_instrument
            except Exception:
                pass
            safe_title = sanitize_title(title)
            safe_instrument = sanitize_title(actual_instrument)
            safe_key = sanitize_title(actual_key)
            idx_str = str(idx + 1).zfill(2)
            filename = f"{idx_str}_{safe_title}_{safe_instrument}_{safe_key}.pdf"
            try:
                pdf_path = assemble_pdf.invoke({
                    "image_dir": upscaled_dir,
                    "output_pdf": filename,
                })
            except Exception:
                pdf_path = None
            if pdf_path and isinstance(pdf_path, str):
                # Copy into orders folder
                try:
                    orders_root = os.path.join(os.getcwd(), "output", "orders", date_folder)
                    os.makedirs(orders_root, exist_ok=True)
                    import shutil
                    shutil.copyfile(pdf_path, os.path.join(orders_root, filename))
                except Exception:
                    pass
        final_pdfs.append(pdf_path if pdf_path and isinstance(pdf_path, str) else None)
    # Clean up temporary directories recorded during scraping
    try:
        import shutil
        for tmp_dir in list(TEMP_DIRS):
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        TEMP_DIRS.clear()
    except Exception:
        pass
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
    # Add new pipeline stages: scrape all songs, remove watermarks, upscale, assemble
    graph.add_node("scraper", scrape_songs_node)
    graph.add_node("watermark_removal", remove_watermarks_node)
    graph.add_node("upscaler", upscale_songs_node)
    graph.add_node("assembler", assemble_pdfs_node)
    # Wire the nodes in sequence: parse -> extract -> scrape -> remove -> upscale -> assemble
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "extractor")
    graph.add_edge("extractor", "scraper")
    graph.add_edge("scraper", "watermark_removal")
    graph.add_edge("watermark_removal", "upscaler")
    graph.add_edge("upscaler", "assembler")
    graph.add_edge("assembler", END)
    return graph.compile()


# Expose a compiled graph instance for LangGraph CLI discovery.
graph = compile_graph()
