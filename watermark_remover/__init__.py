"""
Top-level package for the Watermark Remover project.

This package originally provided a PyQt-based GUI application for
scraping sheet music from the web, removing watermarks using a UNet
model and upscaling images via a VDSR model.  In addition to the GUI
components, the repository now includes an `agent` subpackage which
contains tools and a LangGraph definition for orchestrating the
watermark removal pipeline in an agentic fashion.  See
`watermark_remover/agent/graph.py` for the workflow and
`watermark_remover/agent/tools.py` for the callable functions
used in the workflow.
"""

# Re-export inference models for convenience when importing from the
# top-level package.  Note: these imports are optional and will
# succeed only if `torch` and related dependencies are installed.
try:
    from .inference.model_functions import UNet, VDSR, PIL_to_tensor, tensor_to_PIL, load_best_model  # noqa: F401
except Exception:
    # Optional dependencies may not be available during installation or in
    # constrained environments.  In those cases, the agent code will still
    # function because it lazily imports these within the tools.
    UNet = VDSR = PIL_to_tensor = tensor_to_PIL = load_best_model = None  # type: ignore