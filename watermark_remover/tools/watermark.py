"""Watermark removal tool using a U-Net model."""

from __future__ import annotations

import os
import time
from typing import Optional

from langchain_core.tools import tool
from PIL import Image

from .scrape import logger

try:
    import torch  # type: ignore
    from watermark_remover.inference.model_functions import (
        UNet,
        PIL_to_tensor,
        tensor_to_PIL,
        load_best_model,
    )
except Exception as e:  # broad except to handle ImportError and others
    UNet = PIL_to_tensor = tensor_to_PIL = load_best_model = None  # type: ignore
    torch = None  # type: ignore
    _import_error = e
else:
    _import_error = None


@tool
def remove_watermark(
    input_dir: str,
    model_dir: str = "models/Watermark_Removal",
    output_dir: str = "processed",
) -> str:
    """Remove watermarks from all images in ``input_dir`` using a UNet model.

    Parameters
    ----------
    input_dir : str
        Directory containing input images.
    model_dir : str
        Directory containing UNet checkpoint files.  The most recently
        trained model will be selected based on filename ordering.
    output_dir : str
        Directory to which watermark‑free images are saved.

    Returns
    -------
    str
        Path to the directory containing watermark‑free images.

    Notes
    -----
    If PyTorch or the UNet implementation cannot be imported, this function
    will raise an ImportError when called.  If the model directory
    contains no checkpoints, the UNet model will run with randomly
    initialised weights, which will not produce meaningful results but
    allows the pipeline to run end‑to‑end without bundling large model
    weights in the repository.
    """
    start = time.perf_counter()
    if _import_error is not None:
        raise ImportError(
            f"Cannot import UNet and related utilities: {_import_error}. "
            "Ensure torch and watermark_remover dependencies are installed."
        )
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    # Determine the output directory.  If the caller provides a value
    # other than the default, honour it.  Otherwise, derive a
    # ``2_watermark_removed`` sibling inside the same run directory as
    # the input.  This ensures that all intermediate artefacts live
    # alongside the original images for this run.
    if output_dir == "processed" or not output_dir:
        parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
        output_dir = os.path.join(parent_dir, "2_watermark_removed")
    os.makedirs(output_dir, exist_ok=True)
    # List images to process
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not images:
        raise RuntimeError(f"WMR: no images found in {input_dir}")
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    # Load the best checkpoint if available.  If loading fails, the model
    # will continue with random weights so the pipeline still runs.
    try:
        load_best_model(model, model_dir)  # type: ignore[misc]
    except Exception:
        logger.warning("WMR: failed to load checkpoints from %s; using random weights", model_dir)
    model.eval()
    processed_dir = output_dir
    for fname in images:
        inp_path = os.path.join(input_dir, fname)
        out_path = os.path.join(processed_dir, fname)
        with torch.no_grad():
            # Load and convert the image to a tensor.  If the tensor has three
            # channels, explicitly convert the image to grayscale first to
            # satisfy the UNet input shape (1 channel).  Some downloaded
            # previews are RGB even when PIL_to_tensor normally converts
            # images to grayscale.  Converting here ensures the network
            # receives a 1×H×W tensor and avoids "expected 1 channel but got 3"
            # runtime errors.
            tensor = PIL_to_tensor(inp_path)
            # tensor shape is (C, H, W)
            if tensor.dim() == 3 and tensor.shape[0] != 1:
                try:
                    im = Image.open(inp_path).convert("L")
                    # save temporary grayscale image
                    tmp_path = inp_path + ".gray.tmp"
                    im.save(tmp_path)
                    tensor = PIL_to_tensor(tmp_path)
                    os.remove(tmp_path)
                except Exception:
                    # fallback: average channels
                    tensor = tensor.mean(dim=0, keepdim=True)
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            img = tensor_to_PIL(output.squeeze(0).cpu())
        os.makedirs(processed_dir, exist_ok=True)
        img.save(out_path)
        logger.info("WMR: processed %s -> %s", inp_path, out_path)
    logger.info("WMR completed in %.3fs", time.perf_counter() - start)
    # Do not track the processed directory for cleanup.  We intentionally
    # preserve each intermediate stage under its timestamped run
    # directory for debugging and reproducibility.
    return processed_dir


