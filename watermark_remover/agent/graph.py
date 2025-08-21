"""
Graph definition for orchestrating the Watermark Remover pipeline.

This module builds a simple linear workflow using LangGraph's
``StateGraph``.  The graph contains four nodes: ``scraper``,
``watermark_removal``, ``upscaler`` and ``assembler``.  Each node
invokes a corresponding tool defined in ``tools.py`` and stores its
output in the graph's mutable state.  The workflow proceeds from
scraping to watermark removal to upscaling and finally PDF
assembly.

To visualise and interact with this graph using LangGraph Studio,
run ``langgraph dev`` at the repository root.  The configuration in
``langgraph.json`` points the CLI to the compiled graph defined
here.
"""

from __future__ import annotations

from typing import Any, Dict

import os  # Needed for directory existence checks in scraper_node

import os  # Needed for directory existence checks in scraper_node

from langgraph.graph import StateGraph, START, END

from .tools import scrape_music, remove_watermark, upscale_images, assemble_pdf


class PipelineState(Dict[str, Any]):
    """State dictionary passed between graph nodes.

    The keys used in this state are not fixed; nodes insert and
    retrieve values as needed.  In this pipeline the following
    keys are used:

    * ``title``: song title provided by the user.
    * ``instrument``: instrument name provided by the user.
    * ``key``: musical key provided by the user.
    * ``download_path``: directory returned by the ``scrape_music`` tool.
    * ``processed_path``: directory returned by the ``remove_watermark`` tool.
    * ``upscaled_path``: directory returned by the ``upscale_images`` tool.
    * ``final_pdf``: path to the PDF created by the ``assemble_pdf`` tool.
    """


def scraper_node(state: PipelineState) -> PipelineState:
    """Execute the scraping step and update state with the download path.

    This helper extracts user‑provided metadata (title, instrument, key)
    from the existing state, calls the ``scrape_music`` tool via its
    ``invoke`` method and stores the returned directory path under
    ``download_path``.  It always returns a fresh copy of the state
    to ensure that LangGraph registers the change, and it guards
    against the tool returning ``None``.
    """
    # Create a new state dict to avoid in-place mutation issues
    new_state: PipelineState = dict(state)
    # Extract required fields with sensible defaults
    title = state.get("title", "Unknown Title")
    instrument = state.get("instrument", "Unknown Instrument")
    key = state.get("key", "Unknown Key")
    # Invoke the tool with structured input.  Provide the default
    # input_dir explicitly; without it, the tool's default won't be
    # applied when invoked via LangGraph.
    try:
        download_dir = scrape_music.invoke({
            "title": title,
            "instrument": instrument,
            "key": key,
            "input_dir": "data/samples",
        })
    except Exception:
        # In case of any error (e.g. directory missing), fallback to
        # the default sample directory if it exists; otherwise set to None
        download_dir = "data/samples" if os.path.isdir("data/samples") else None
    # Normalise return value to str if possible
    new_state["download_path"] = None
    if download_dir:
        new_state["download_path"] = (
            download_dir if isinstance(download_dir, str) else str(download_dir)
        )
    return new_state


def watermark_removal_node(state: PipelineState) -> PipelineState:
    """Execute the watermark removal step and update state.

    This node consumes ``download_path`` from the state and passes it to
    the ``remove_watermark`` tool.  It writes the resulting directory to
    ``processed_path`` on a new state dict, safeguarding against
    ``None`` values.  If ``download_path`` is missing or invalid, the
    node leaves ``processed_path`` unset, which will trigger a
    subsequent validation error downstream.
    """
    new_state: PipelineState = dict(state)
    input_dir = state.get("download_path")
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


def upscaler_node(state: PipelineState) -> PipelineState:
    """Execute the upscaling step and update state.

    Reads ``processed_path`` from the input state and passes it to
    ``upscale_images.invoke``.  Stores the resulting directory in
    ``upscaled_path`` on a new state dict.  If ``processed_path`` is
    missing or invalid, ``upscaled_path`` will remain ``None``.
    """
    new_state: PipelineState = dict(state)
    input_dir = state.get("processed_path")
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


def assembler_node(state: PipelineState) -> PipelineState:
    """Execute the PDF assembly step and update state with the PDF path.

    Reads ``upscaled_path`` and ``title`` from the state, invokes
    ``assemble_pdf`` with these parameters, and writes the resulting
    PDF path to ``final_pdf`` on a new state dict.  If ``upscaled_path``
    is missing or invalid, ``final_pdf`` will remain ``None``.
    """
    new_state: PipelineState = dict(state)
    image_dir = state.get("upscaled_path")
    title = state.get("title", "output")
    pdf_path = None
    if image_dir:
        try:
            pdf_path = assemble_pdf.invoke({
                "image_dir": image_dir,
                "output_pdf": f"{title}.pdf",
            })
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

    Returns
    -------
    compiled_graph
        A runnable graph that can be invoked directly or passed to
        `langgraph dev` for visualisation.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("scraper", scraper_node)
    graph.add_node("watermark_removal", watermark_removal_node)
    graph.add_node("upscaler", upscaler_node)
    graph.add_node("assembler", assembler_node)

    graph.add_edge(START, "scraper")
    graph.add_edge("scraper", "watermark_removal")
    graph.add_edge("watermark_removal", "upscaler")
    graph.add_edge("upscaler", "assembler")
    graph.add_edge("assembler", END)

    return graph.compile()


# Expose a compiled graph instance for LangGraph CLI discovery.
graph = compile_graph()
