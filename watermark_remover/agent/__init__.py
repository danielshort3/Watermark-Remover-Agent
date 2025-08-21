"""
Agent subpackage for the Watermark Remover project.

This package exposes a set of tools and a LangGraph definition to
enable multi‑agent orchestration of the watermark remover pipeline.

The tools in `tools.py` wrap the core functionality for downloading
sheet music (currently implemented as a stub), removing watermarks
with a UNet model, upscaling with a VDSR model and assembling the
processed images into a PDF.  The graph defined in `graph.py`
connects these steps into a linear workflow.  When run under
`langgraph dev`, the graph can be visualised in LangGraph Studio and
each step executed sequentially.
"""

from .tools import scrape_music, remove_watermark, upscale_images, assemble_pdf  # noqa: F401
from .graph import graph  # noqa: F401
