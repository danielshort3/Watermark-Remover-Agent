"""
Agent subpackage for the Watermark Remover project.

This package exposes a set of tools and agent helpers for the
watermark remover pipeline.  The tools in ``tools.py`` wrap the core
functionality for downloading sheet music (currently implemented as a
stub), removing watermarks with a UNet model, upscaling with a VDSR
model and assembling the processed images into a PDF.  The graph
defined in ``graph.py`` connects these steps into a linear workflow
using a minimal stub of LangGraph.  The original LLM‑driven graph
defined in ``graph_ollama.py`` has been replaced by a direct
interface: use :func:`watermark_remover.agent.graph_ollama.run_instruction` to
execute a natural‑language task with the Ollama‑powered LangChain
agent or call :func:`get_ollama_agent` from ``ollama_agent.py`` to
obtain a reusable agent instance.
"""

# Lazily import optional submodules.  Some modules (e.g. tools) depend on
# external libraries such as ``langchain`` that may not be installed in
# all environments.  Wrapping these imports in a try/except allows this
# package to be imported even when optional dependencies are missing.
try:
    from .tools import scrape_music, remove_watermark, upscale_images, assemble_pdf  # type: ignore
except Exception:
    # When dependencies like langchain are not available, these
    # functions will be unavailable.  They can still be imported
    # directly from watermark_remover.agent.tools when running in a
    # fully configured environment.
    pass

# graph and get_ollama_agent may depend on optional dependencies.  Wrap
# their imports in try/except so that importing this package does not
# raise errors when optional dependencies are missing.  These symbols
# can still be imported directly from their submodules when the
# environment is fully configured.
try:
    from .graph import graph  # type: ignore
except Exception:
    # The pipeline graph requires langchain in tools.py.  Skip import if
    # dependencies are unavailable.
    pass

try:
    from .ollama_agent import get_ollama_agent  # type: ignore
except Exception:
    # get_ollama_agent depends on langchain and langchain_ollama.  Skip
    # import if they are not installed.
    pass

# Expose the run_instruction helper at the package level.  If optional
# dependencies are missing this import will fail silently.
try:
    from .graph_ollama import run_instruction  # type: ignore
except Exception:
    pass