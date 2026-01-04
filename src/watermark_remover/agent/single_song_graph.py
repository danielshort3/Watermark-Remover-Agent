"""
Multi-agent graph definition for the Watermark Remover pipeline using LangGraph.

This module defines a graph that parses a natural-language instruction into
structured metadata (title, instrument and key) using a single call to an
Ollama-backed chat model.  It then executes the scraping, watermark removal,
upscaling and PDF assembly tools in sequence, updating the mutable state at
each step.  By sharing a single ChatOllama instance across all nodes, the
graph minimises round‑trips to the model and allows the entire pipeline to be
managed via LangGraph Studio.

To run this graph locally with a GUI:
  1. Build and run the Docker container as described in the repository README.
  2. Visit the LangGraph Studio URL printed by the container on startup
     (e.g. https://smith.langchain.com/studio/?baseUrl=http://localhost:2024).
  3. Use the Studio interface to execute the `watermark-remover` graph with
     a `user_input` field (e.g. {"user_input": "Download Fur Elise for French Horn in G"}).

The graph expects a dictionary input with at least the key ``user_input``.
Additional optional fields (``title``, ``instrument``, ``key``, ``input_dir``)
may be provided to override or supplement the metadata extracted by the model.
"""
from __future__ import annotations

# Note: moved under src/ layout for consistency.

import os
import json
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
try:
    # Command is only needed for dynamic routing.  Importing it is optional
    from langgraph.graph import Command  # type: ignore
except Exception:
    Command = None  # type: ignore

# Attempt to import ChatOllama; if unavailable at import time the module will
# still load and the error will be raised at runtime when the parser node runs.
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)
try:
    from watermark_remover.agent.prompts import build_single_song_parser_prompt, log_prompt
except Exception:
    build_single_song_parser_prompt = None  # type: ignore
    log_prompt = None  # type: ignore

# Regular expression utilities for fallback parsing.
import re

# This module uses plain Python dictionaries to represent the mutable state
# passed between nodes in the pipeline.  We avoid using ``typing.Dict`` in
# function annotations because LangGraph infers state types from these
# annotations.  The Studio attempts to instantiate the type specified on
# the state channel, and ``typing.Dict`` cannot be instantiated directly
# (leading to ``TypeError: Type Dict cannot be instantiated; use dict() instead``).
# By annotating functions with ``dict`` instead of ``Dict[...]``, we ensure
# that LangGraph uses ``dict`` as the concrete type of the root state and
# therefore can construct an empty state for visualisation and debugging.
PipelineState = dict

# Module-level variable to cache the LLM so that it is created only once.
_llm: Any = None


def _get_llm() -> Any:
    """Create or return a cached ChatOllama instance.

    The model name, base URL and keep-alive settings are taken from the
    environment variables OLLAMA_MODEL, OLLAMA_URL and OLLAMA_KEEP_ALIVE,
    respectively.  Defaults are provided for convenience.  If ChatOllama is
    not installed, an ImportError is raised at runtime.
    """
    global _llm
    if _llm is not None:
        return _llm
    if ChatOllama is None:
        raise ImportError(
            "ChatOllama is not available; install langchain-ollama to use the parser node."
        )
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
    # Temperature is set to zero for deterministic JSON extraction.
    _llm = ChatOllama(
        model=model, base_url=base_url, keep_alive=keep_alive, temperature=0.0
    )
    return _llm


def parser_node(state: dict) -> dict:
    """Parse the user_input into title, instrument and key using the LLM.

    This node updates the state in place with 'title', 'instrument' and 'key'
    fields if they are absent.  It uses a single LLM call, making this the only
    LLM invocation in the entire pipeline.  If parsing fails, the fields are
    left unchanged and the pipeline proceeds with any defaults provided.
    """
    # Make a copy of state to avoid in-place mutation.
    new_state: PipelineState = dict(state)
    # Only parse if any of the metadata fields are missing.
    if all(new_state.get(k) for k in ("title", "instrument", "key")):
        return new_state
    # Ensure there is a user_input to parse.
    instruction = new_state.get("user_input", "")
    if not instruction:
        return new_state
    llm = _get_llm()
    # Formulate a prompt instructing the model to return JSON only.
    prompt = build_single_song_parser_prompt(instruction) if build_single_song_parser_prompt else (
        "You are a JSON API that extracts structured metadata from a natural-language instruction about downloading sheet music. "
        f"Instruction: {instruction}\n\nJSON:"
    )
    try:
        if log_prompt:
            log_prompt("single_song_parser", prompt)
    except Exception:
        pass
    try:
        # ChatOllama's invoke method may return a message object or raw string.
        response = llm.invoke(prompt)  # type: ignore[no-untyped-call]
        # Extract the text content if a message object is returned.
        if hasattr(response, "content"):
            text = response.content  # type: ignore[assignment]
        else:
            text = response  # type: ignore[assignment]
        try:
            if log_prompt:
                log_prompt("single_song_parser", prompt)
            if log_prompt:
                # reuse function for response as well
                from watermark_remover.agent.prompts import log_llm_response  # type: ignore
                log_llm_response("single_song_parser", text)
        except Exception:
            pass
        # Attempt to parse JSON; if this fails, we will fall back to regex.
        meta = json.loads(text)
    except Exception:
        meta = None
    # If meta is a dict, update state with extracted fields; otherwise fall back.
    if isinstance(meta, dict):
        for key in ("title", "instrument", "key"):
            if key not in new_state or not new_state.get(key):
                val = meta.get(key)
                if isinstance(val, str):
                    new_state[key] = val.strip()
    else:
        # Fallback heuristic parsing: extract title, instrument and key from the instruction.
        extracted = _heuristic_parse(instruction)
        for key in ("title", "instrument", "key"):
            if key not in new_state or not new_state.get(key):
                val = extracted.get(key)
                if isinstance(val, str):
                    new_state[key] = val
    return new_state


def _heuristic_parse(text: str) -> Dict[str, str]:
    """Simple heuristic parser for title, instrument and key.

    This function extracts:

    * title: any substring enclosed in single or double quotes
    * instrument: the word(s) following "for " up to the next " in "
    * key: the word following " in " up to the next comma or end of string

    The returned values may be empty strings if no match is found.
    """
    result: Dict[str, str] = {"title": "", "instrument": "", "key": ""}
    # Find quoted title
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    if m:
        result["title"] = m.group(1).strip()
    # Find instrument after "for " and before " in " or comma
    m2 = re.search(r"for\s+([^,]+?)(?:\s+in\b|,|$)", text, re.IGNORECASE)
    if m2:
        result["instrument"] = m2.group(1).strip()
    # Find key after " in " and before comma
    m3 = re.search(r" in\s+([A-G][b#]?)(?:[,\s]|$)", text, re.IGNORECASE)
    if m3:
        result["key"] = m3.group(1).strip()
    return result


def scraper_node(state: dict) -> dict:
    """Execute the scraping step and update state with the download path."""
    new_state: PipelineState = dict(state)
    # Determine which metadata to use.
    title = new_state.get("title", "Unknown Title")
    instrument = new_state.get("instrument", "Unknown Instrument")
    key = new_state.get("key", "Unknown Key")
    user_input_dir = new_state.get("input_dir", "data/samples")
    try:
        download_dir = scrape_music.invoke(
            {
                "title": title,
                "instrument": instrument,
                "key": key,
                "input_dir": user_input_dir,
            }
        )
    except Exception:
        download_dir = None
    new_state["download_path"] = None
    if download_dir:
        new_state["download_path"] = (
            download_dir if isinstance(download_dir, str) else str(download_dir)
        )
    return new_state


def watermark_removal_node(state: dict) -> dict:
    """Execute the watermark removal step and update state."""
    new_state: PipelineState = dict(state)
    input_dir = new_state.get("download_path")
    processed_dir = None
    if input_dir:
        try:
            processed_dir = remove_watermark.invoke({"input_dir": input_dir})
        except Exception:
            processed_dir = None
    new_state["processed_path"] = None
    if processed_dir:
        new_state["processed_path"] = (
            processed_dir if isinstance(processed_dir, str) else str(processed_dir)
        )
    return new_state


def upscaler_node(state: dict) -> dict:
    """Execute the upscaling step and update state."""
    new_state: PipelineState = dict(state)
    input_dir = new_state.get("processed_path")
    upscaled_dir = None
    if input_dir:
        try:
            upscaled_dir = upscale_images.invoke({"input_dir": input_dir})
        except Exception:
            upscaled_dir = None
    new_state["upscaled_path"] = None
    if upscaled_dir:
        new_state["upscaled_path"] = (
            upscaled_dir if isinstance(upscaled_dir, str) else str(upscaled_dir)
        )
    return new_state


def assembler_node(state: dict) -> dict:
    """Execute the PDF assembly step and update state with the final PDF path."""
    new_state: PipelineState = dict(state)
    image_dir = new_state.get("upscaled_path")
    title = new_state.get("title", "output")
    pdf_path = None
    if image_dir:
        try:
            pdf_path = assemble_pdf.invoke(
                {"image_dir": image_dir, "output_pdf": f"{title}.pdf"}
            )
        except Exception:
            pdf_path = None
    new_state["final_pdf"] = None
    if pdf_path:
        new_state["final_pdf"] = (
            pdf_path if isinstance(pdf_path, str) else str(pdf_path)
        )
    return new_state


def compile_graph() -> Any:
    """Construct and compile the LangGraph pipeline graph.

    The compiled graph can be passed to langgraph.dev to expose a local API
    and GUI for interactive use.  The graph expects an initial state with a
    ``user_input`` string and optional ``title``, ``instrument`` and ``key``
    overrides.
    """
    # The root state type must be a concrete ``dict``.  Passing ``PipelineState``
    # (a ``typing.Dict`` alias) to StateGraph leads to a ``TypeError`` when
    # LangGraph Studio tries to render the graph.  Using ``dict`` here tells
    # LangGraph how to instantiate an empty state when visualizing the graph.
    graph = StateGraph(dict)
    # Add nodes in the order they should run.
    graph.add_node("parser", parser_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("watermark_removal", watermark_removal_node)
    graph.add_node("upscaler", upscaler_node)
    graph.add_node("assembler", assembler_node)

    # Wire the edges to run sequentially.
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "scraper")
    graph.add_edge("scraper", "watermark_removal")
    graph.add_edge("watermark_removal", "upscaler")
    graph.add_edge("upscaler", "assembler")
    graph.add_edge("assembler", END)

    return graph.compile()


# Expose a compiled graph instance for LangGraph CLI discovery.
graph = compile_graph()
