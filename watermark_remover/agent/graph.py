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

from langgraph.graph import StateGraph, START, END

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)


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
    """Execute the scraping step and update state with the download path."""
    title = state.get("title", "Unknown Title")
    instrument = state.get("instrument", "Unknown Instrument")
    key = state.get("key", "Unknown Key")
    download_dir = scrape_music(title=title, instrument=instrument, key=key)
    state["download_path"] = download_dir
    return state


def watermark_removal_node(state: PipelineState) -> PipelineState:
    """Execute the watermark removal step and update state."""
    input_dir = state.get("download_path")
    processed_dir = remove_watermark(input_dir=input_dir)
    state["processed_path"] = processed_dir
    return state


def upscaler_node(state: PipelineState) -> PipelineState:
    """Execute the upscaling step and update state."""
    input_dir = state.get("processed_path")
    upscaled_dir = upscale_images(input_dir=input_dir)
    state["upscaled_path"] = upscaled_dir
    return state


def assembler_node(state: PipelineState) -> PipelineState:
    """Execute the PDF assembly step and update state with the PDF path."""
    image_dir = state.get("upscaled_path")
    title = state.get("title", "output")
    pdf_path = assemble_pdf(image_dir=image_dir, output_pdf=f"{title}.pdf")
    state["final_pdf"] = pdf_path
    return state


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
