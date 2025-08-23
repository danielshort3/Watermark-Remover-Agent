"""Inference subpackage for the Watermark Remover project.

This package houses implementations of the neural network models used
throughout the project.  The primary module, :mod:`model_functions`,
contains definitions for both the VDSR super‑resolution network and a
U‑Net for watermark removal, along with a handful of utility
functions.  By isolating these classes and helpers into a dedicated
subpackage, the rest of the codebase can lazily import heavy
dependencies (like PyTorch) only when needed.

The :mod:`model_functions` module also exposes functions to load the
most recent model checkpoint from a directory (with safeguards for
missing or invalid files) and to convert between PIL images and
torch tensors.
"""

from .model_functions import (
    VDSR,
    UNet,
    PIL_to_tensor,
    tensor_to_PIL,
    load_best_model,
    load_model,
    PerceptualLoss,
    CombinedLoss,
)  # noqa: F401