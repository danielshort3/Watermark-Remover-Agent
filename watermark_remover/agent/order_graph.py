"""Order of worship graph for processing multiple songs.

This module defines a simple graph that parses an order‑of‑worship PDF
and processes each song using the existing Watermark Remover pipeline
defined in ``watermark_remover.agent.multi_agent_graph``.  Given a
PDF path and an instrument name, the graph extracts song titles,
artists and keys from the PDF using basic heuristics and then runs
scraping, watermark removal, upscaling and PDF assembly for each
song in sequence.  The output PDFs are organised under the
``output/music`` directory as in the main pipeline and the
intermediate artefacts are stored in the timestamped logs folder.

Usage:

    from watermark_remover.agent.order_graph import run_order_graph
    run_order_graph(pdf_path="input/orders/my_order.pdf", instrument="Piano")

The PDF must be placed in the ``input/orders`` directory relative to
the repository root.  The instrument argument is passed through to
``scrape_music`` to select the appropriate arrangement.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
import shutil
import datetime

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

from langgraph.graph import StateGraph, START, END

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)

from watermark_remover.agent.graph import graph as _wmr_graph

def parse_order_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Parse an order‑of‑worship PDF into a list of songs.

    The parser attempts to extract song metadata (title, artist and
    key) from each line of the PDF.  It supports a variety of common
    formats, including:

    * ``Title by Artist in Key``
    * ``Title – Artist – Key`` (em or en dash, or hyphen)
    * ``Title (Artist) – Key``
    * ``Title – Key`` (no artist)

    The key may optionally be prefaced by words like "in", "key of",
    "Key" or "in the key of".  If a line does not contain a
    recognisable key (A–G with optional sharp or flat), it is ignored.
    Lines that clearly do not correspond to music (e.g. headings,
    numbers, prayers) should be filtered out by the caller or by
    examining the extracted metadata.

    The implementation first attempts to use ``pdfplumber`` to extract
    text.  If that library is unavailable or cannot read the PDF, it
    falls back to the system ``pdftotext`` command (if present) to
    obtain a plain‑text representation of the document.  If both
    methods fail, an empty list is returned.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to parse.

    Returns
    -------
    list of dict
        A list of dictionaries with keys ``title``, ``artist`` and
        ``key`` for each recognised song.  The ``artist`` field will
        be set to ``"unknown"`` if no artist is found.
    """
    songs: List[Dict[str, str]] = []
    # Early exit if file does not exist
    if not os.path.isfile(pdf_path):
        return songs
    # Helper to normalise extracted metadata
    def _append_song(title: str, artist: str, key: str) -> None:
        title = title.strip()
        artist = artist.strip() or "unknown"
        key = key.strip()
        if not title or not key:
            return
        songs.append({"title": title, "artist": artist, "key": key})

    # Attempt to read the PDF using pdfplumber if available
    text_lines: List[str] = []
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_lines.extend(page_text.split("\n"))
        except Exception:
            # Fall through to pdftotext
            text_lines = []
    # If no lines extracted and pdftotext is available, use it as a fallback
    if not text_lines:
        try:
            import subprocess, shlex
            cmd = ["pdftotext", pdf_path, "-"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                text_lines = proc.stdout.splitlines()
        except Exception:
            text_lines = []
    # Compile regular expressions once for efficiency
    # Pattern for lines like "Title by Artist in Key".  The key may be
    # prefaced by words like "in", "key of", "key", etc.
    pattern_bracket = re.compile(
        r"^(?P<title>.*?)\s*\[\s*(?P<artist>.*?)\s+in\s+(?:the\s+key\s+of\s+|key\s+of\s+|key\s+)?(?P<key>[A-G][b#]?)\s*\]$",
        re.IGNORECASE,
    )
    pattern_by = re.compile(
        r"^(?P<title>.*?)\s+by\s+(?P<artist>.+?)\s+(?:in\s+(?:the\s+key\s+of\s+|the\s+key\s+|key\s+of\s+|key\s+)?|key\s+of\s+|key\s+|in\s+)?(?P<key>[A-G][b#]?)\b",
        re.IGNORECASE,
    )
    # Pattern for lines like "Title (Artist) – Key" or "Title (Artist) - Key"
    pattern_paren = re.compile(
        r"^(?P<title>.*?)\s*\((?P<artist>.*?)\)\s*[–-]\s*(?P<key>[A-G][b#]?)",
        re.IGNORECASE,
    )
    # Pattern for lines separated by dashes where the last token contains the key
    splitter = re.compile(r"\s*[–—-]\s*")
    key_re = re.compile(r"([A-G][b#]?)\b")
    for raw_line in text_lines:
        line = raw_line.strip()
        if not line:
            continue
        # Remove a leading time stamp (e.g. "3:30 " or "10:45 ") if present.  Many
        # order‑of‑worship documents prefix song titles with a clock time
        # indicating when the song occurs.  This interferes with parsing,
        # so strip it off before applying other patterns.  The pattern
        # matches one or two digits, a colon, two digits, optional AM/PM,
        # and any following whitespace.
        line = re.sub(r"^\s*\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\s+", "", line)
        # Try the bracketed pattern first, e.g. "Title [ Artist in Key ]".
        mb = pattern_bracket.match(line)
        if mb:
            title = mb.group("title").strip()
            artist = mb.group("artist").strip()
            key = mb.group("key").strip()
            # Treat generic arrangements as unknown artists
            if artist.lower().startswith("default arrangement"):
                artist = "unknown"
            _append_song(title, artist, key)
            continue
        # Try the "by" pattern
        m = pattern_by.match(line)
        if m:
            _append_song(m.group("title"), m.group("artist"), m.group("key"))
            continue
        # Try the parenthetical pattern
        m2 = pattern_paren.match(line)
        if m2:
            _append_song(m2.group("title"), m2.group("artist"), m2.group("key"))
            continue
        # Try splitting on dashes (em, en, hyphen) and parse the last element for key
        parts = splitter.split(line)
        if len(parts) >= 2:
            last = parts[-1]
            km = key_re.search(last)
            if km:
                key = km.group(1)
                # Determine artist and title based on number of parts
                if len(parts) >= 3:
                    artist = parts[-2].strip()
                    title = " - ".join([p.strip() for p in parts[:-2]])
                else:
                    artist = "unknown"
                    title = parts[0].strip()
                _append_song(title, artist, key)
                continue
        # As a final heuristic, look for a standalone key within parentheses at end
        # e.g. "Song Name (Key)" or "Song Name Key"
        paren_key = re.search(r"\((?P<key>[A-G][b#]?)\)$", line)
        if paren_key:
            key = paren_key.group("key")
            # Remove the parenthesised key from the line to extract title
            title = re.sub(r"\([A-G][b#]?\)$", "", line).strip()
            _append_song(title, "unknown", key)
            continue
        # Next, look for the key preceded by common phrases like "Key",
        # "Key of", "in" etc.  This covers lines like "Title – Key of D".
        m3 = re.search(r"(?:in\s+(?:the\s+key\s+of\s+|key\s+of\s+|key\s+)?)|(?:key\s+of\s+)|(?:key\s+)", line, re.IGNORECASE)
        if m3:
            # Extract key by searching for A-G followed by optional accidental
            km2 = key_re.search(line[m3.end():])
            if km2:
                key = km2.group(1)
                # Everything before the phrase is treated as title and artist
                prefix = line[: m3.start()].strip()
                # Attempt to separate title and artist by dash or parentheses
                parts2 = splitter.split(prefix)
                if len(parts2) >= 2:
                    artist = parts2[-1].strip()
                    title = " - ".join([p.strip() for p in parts2[:-1]])
                else:
                    title = prefix
                    artist = "unknown"
                _append_song(title, artist, key)
                continue
    return songs


def run_order_graph(pdf_path: str, instrument: str) -> None:
    """Execute the order of worship graph.

    This function orchestrates the parsing of the provided PDF and
    processes each extracted song sequentially using the existing
    watermark remover pipeline.  The instrument is passed through to
    the scraper to determine the arrangement.

    Parameters
    ----------
    pdf_path : str
        Path to the order of worship PDF (relative to repository root).
    instrument : str
        The instrument for which the arrangements should be downloaded
        (e.g. "piano", "violin").
    """
    # Parse the PDF
    songs = parse_order_pdf(pdf_path)
    if not songs:
        print(f"No songs found in {pdf_path}")
        return
    # Prepare an orders directory for this run.  Each run uses the
    # current timestamp unless RUN_TS is already set (e.g. by tools.py).
    run_ts = os.environ.get("RUN_TS") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orders_root = os.path.join(os.getcwd(), "output", "orders", run_ts)
    os.makedirs(orders_root, exist_ok=True)
    counter = 1
    for song in songs:
        title = song.get("title", "")
        artist = song.get("artist", "")
        key = song.get("key", "")
        # Invoke scraping.  Note: scrape_music will set SCRAPE_METADATA
        # accordingly.  We pass the instrument argument through.
        download_dir = scrape_music.invoke(
            {"title": title, "instrument": instrument, "key": key, "input_dir": "data/samples"}
        )
        if not download_dir:
            print(f"Failed to download music for {title} ({artist}) in {key}")
            continue
        # Run watermark removal
        wmr_dir = remove_watermark.invoke({"input_dir": download_dir})
        # Run upscaling
        us_dir = upscale_images.invoke({"input_dir": wmr_dir})
        # Assemble PDF
        pdf_path = assemble_pdf.invoke({"image_dir": us_dir})
        # Copy the resulting PDF into the orders directory with a numerical prefix
        try:
            if isinstance(pdf_path, str) and os.path.isfile(pdf_path):
                dest_name = f"{counter}_" + os.path.basename(pdf_path)
                dest_path = os.path.join(orders_root, dest_name)
                shutil.copyfile(pdf_path, dest_path)
                counter += 1
        except Exception:
            pass


def build_order_state_graph() -> StateGraph:
    """Construct a simple LangGraph for processing an order of worship.

    The returned graph expects a state dict with keys ``pdf_path`` and
    ``instrument``.  It will parse the PDF and iterate through the
    extracted songs, invoking the existing pipeline tools for each one.
    """
    sg: StateGraph = StateGraph(dict)

    def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        pdf_path = new_state.get("pdf_path")
        if pdf_path:
            songs = parse_order_pdf(pdf_path)
            new_state["songs"] = songs
        else:
            new_state["songs"] = []
        return new_state

    def process_node(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        songs = new_state.get("songs", []) or []
        instrument_name = new_state.get("instrument", "")
        for song in songs:
            title = song.get("title", "")
            artist = song.get("artist", "")
            key = song.get("key", "")
            try:
                download_dir = scrape_music.invoke(
                    {"title": title, "instrument": instrument_name, "key": key, "input_dir": "data/samples"}
                )
                if not download_dir:
                    continue
                wmr_dir = remove_watermark.invoke({"input_dir": download_dir})
                us_dir = upscale_images.invoke({"input_dir": wmr_dir})
                assemble_pdf.invoke({"image_dir": us_dir})
            except Exception:
                continue
        return new_state

    sg.add_node("parse", parser_node)
    sg.add_node("process", process_node)
    # Edges
    sg.set_entry_point("parse")
    sg.add_edge("parse", "process")
    sg.add_edge("process", END)
    return sg


# ---------------------------------------------------------------------------
# LangGraph integration
#
# The functions below integrate the order of worship pipeline with LangGraph
# so that it can be executed via the Studio or CLI.  The graph expects
# ``user_input`` describing the PDF and instrument, and will then parse
# and process each song accordingly.

def _parse_instruction(instruction: str) -> Dict[str, str]:
    """Extract the PDF filename and instrument from a natural language instruction.

    The parser looks for a quoted PDF filename and the word following
    ``for`` to identify the instrument.  If either element is missing,
    empty strings are returned.

    Parameters
    ----------
    instruction : str
        Natural language instruction (e.g. from ``user_input``).

    Returns
    -------
    dict
        A dictionary with keys ``pdf_path`` and ``instrument``.
    """
    result: Dict[str, str] = {"pdf_path": "", "instrument": ""}
    if not instruction:
        return result
    # Find a quoted filename ending with .pdf
    m = re.search(r"['\"]([^'\"]+\.pdf)['\"]", instruction, re.IGNORECASE)
    if m:
        result["pdf_path"] = m.group(1).strip()
    # Look for "for <instrument>" in the instruction
    m2 = re.search(r"\bfor\s+([A-Za-z ]+?)(?:\s+and|\s*$)", instruction, re.IGNORECASE)
    if m2:
        result["instrument"] = m2.group(1).strip()
    return result


def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the user's instruction into pdf_path and instrument fields."""
    new_state = dict(state)
    instruction = new_state.get("user_input", "")
    parsed = _parse_instruction(instruction)
    # Prefix the pdf_path with the orders directory if a bare filename is provided
    pdf_path = parsed.get("pdf_path", "")
    if pdf_path and not os.path.isabs(pdf_path):
        pdf_path = os.path.join("input", "orders", pdf_path)
    new_state["pdf_path"] = pdf_path
    new_state["instrument"] = parsed.get("instrument", "")
    return new_state


def pdf_parse_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load the PDF and extract songs using ``parse_order_pdf``."""
    new_state = dict(state)
    pdf_path = new_state.get("pdf_path", "")
    songs = parse_order_pdf(pdf_path) if pdf_path else []
    new_state["songs"] = songs
    return new_state


def process_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process each song sequentially using the existing tools."""
    new_state = dict(state)
    songs = new_state.get("songs", []) or []
    instrument_name = new_state.get("instrument", "")
    # Prepare an orders directory for this run based on timestamp.  Use
    # the existing RUN_TS if defined to keep consistency with the
    # logging structure.  Otherwise, generate a new timestamp.
    run_ts = os.environ.get("RUN_TS") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orders_root = os.path.join(os.getcwd(), "output", "orders", run_ts)
    os.makedirs(orders_root, exist_ok=True)
    counter = 1
    for song in songs:
        title = song.get("title", "")
        key = song.get("key", "")
        try:
            download_dir = scrape_music.invoke(
                {"title": title, "instrument": instrument_name, "key": key, "input_dir": "data/samples"}
            )
            if not download_dir:
                continue
            wmr_dir = remove_watermark.invoke({"input_dir": download_dir})
            us_dir = upscale_images.invoke({"input_dir": wmr_dir})
            pdf_path = assemble_pdf.invoke({"image_dir": us_dir})
            # Copy PDF to orders folder with numerical prefix
            if isinstance(pdf_path, str) and os.path.isfile(pdf_path):
                dest_name = f"{counter}_" + os.path.basename(pdf_path)
                dest_path = os.path.join(orders_root, dest_name)
                try:
                    shutil.copyfile(pdf_path, dest_path)
                    counter += 1
                except Exception:
                    pass
        except Exception:
            continue
    return new_state

def process_songs(state: Dict[str, Any]) -> Dict[str, Any]:
    new_state = dict(state)
    songs: List[Dict[str, Any]] = new_state.get("songs", []) or []

    # Pull the pdf_path and instrument from the incoming state.
    pdf_path: str = new_state.get("pdf_path", "") or ""
    instrument: str = new_state.get("instrument", "") or ""

    results: List[Dict[str, Any]] = []
    for song in songs:
        title = song.get("title") or None
        if not title:
            continue

        sub_state = {
            "pdf_path": pdf_path,
            "instrument": instrument,
            "title": title,
            "artist": song.get("artist"),
            "key": song.get("key"),
        }
        final_pdf: Optional[str] = None
        try:
            result_state = _wmr_graph.invoke(sub_state)  # type: ignore
            if isinstance(result_state, dict):
                final_pdf = result_state.get("final_pdf")
        except Exception:
            final_pdf = None

        results.append({"title": title, "final_pdf": final_pdf})

    new_state["song_results"] = results
    return new_state

def compile_graph() -> Any:
    sg = StateGraph(dict)
    sg.add_node("parse_instruction", parser_node)
    sg.add_node("parse_pdf", pdf_parse_node)
    sg.add_node("process_songs", process_songs)  # not process_node
    sg.add_edge(START, "parse_instruction")
    sg.add_edge("parse_instruction", "parse_pdf")
    sg.add_edge("parse_pdf", "process_songs")
    sg.add_edge("process_songs", END)
    return sg.compile()


# Expose a compiled graph instance for LangGraph CLI discovery.
graph = compile_graph()
