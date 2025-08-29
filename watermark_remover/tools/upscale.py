"""Image upscaling tool using a VDSR network."""

from __future__ import annotations

import os
import time
from typing import Optional

from langchain_core.tools import tool

from .scrape import logger

try:
    import torch  # type: ignore
    from watermark_remover.inference.model_functions import (
        VDSR,
        PIL_to_tensor,
        tensor_to_PIL,
        load_best_model,
    )
except Exception as e:  # broad except to handle ImportError and others
    VDSR = PIL_to_tensor = tensor_to_PIL = load_best_model = None  # type: ignore
    torch = None  # type: ignore
    _import_error = e
else:
    _import_error = None


@tool
def upscale_images(
    input_dir: str,
    model_dir: str = "models/VDSR",
    output_dir: str = "upscaled",
) -> str:
    """Upscale all images in ``input_dir`` using a VDSR model.

    This implementation mirrors the patch‑based algorithm used in the
    original Watermark Remover project.  Each image is first upsampled
    to a fixed size using nearest‑neighbour interpolation and then
    processed in overlapping patches through the VDSR network.  The
    results are stitched back together to form the final high‑resolution
    image.

    Parameters
    ----------
    input_dir : str
        Directory containing watermark‑free images.
    model_dir : str
        Directory containing VDSR checkpoint files.
    output_dir : str
        Directory to which upscaled images are saved.

    Returns
    -------
    str
        Path to the directory containing upscaled images.

    Notes
    -----
    If PyTorch or the VDSR implementation cannot be imported, this function
    will raise an ImportError when called.  If the model directory
    contains no checkpoints, the VDSR model will run with randomly
    initialised weights, which will not produce meaningful results but
    allows the pipeline to run end‑to‑end without bundling large model
    weights in the repository.
    """
    start = time.perf_counter()
    if _import_error is not None:
        raise ImportError(
            f"Cannot import VDSR and related utilities: {_import_error}. "
            "Ensure torch and watermark_remover dependencies are installed."
        )
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    # Determine the output directory.  If the caller provides a value
    # other than the default, honour it.  Otherwise, derive a
    # ``3_upscaled`` sibling inside the same run directory as the input.
    if output_dir == "upscaled" or not output_dir:
        parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
        output_dir = os.path.join(parent_dir, "3_upscaled")
    os.makedirs(output_dir, exist_ok=True)
    images = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]
    if not images:
        raise RuntimeError(f"UPSCALE: no images found in {input_dir}")
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    # Instantiate the VDSR model and load the best checkpoint if available.
    us_model = VDSR().to(device)
    try:
        load_best_model(us_model, model_dir)  # type: ignore[misc]
    except Exception:
        logger.warning("VDSR: failed to load checkpoints from %s; using random weights", model_dir)
    us_model.eval()
    # Define upsample operation to enlarge each image to the canonical size.
    image_base_width, image_base_height = 1700, 2200
    upsample = torch.nn.Upsample(size=(image_base_height, image_base_width), mode="nearest")
    # Patch parameters matching the original implementation
    padding_size = 16
    patch_height = 550
    patch_width = 850
    # Process each image
    for fname in images:
        inp_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        # Convert image to tensor and move to device
        with torch.no_grad():
            tensor = PIL_to_tensor(inp_path)
            tensor = tensor.unsqueeze(0).to(device)
            # Upsample to target size
            wm_output_upscaled = upsample(tensor)
            # Pad so patches at the edges can be processed uniformly
            padding = (padding_size, padding_size, padding_size, padding_size)
            wm_output_upscaled_padded = torch.nn.functional.pad(
                wm_output_upscaled, padding, value=1.0
            )
            # Prepare an output tensor
            us_output = torch.zeros_like(wm_output_upscaled)
            # Slide window over the upscaled image
            for i in range(0, wm_output_upscaled.shape[-2], patch_height):
                for j in range(0, wm_output_upscaled.shape[-1], patch_width):
                    patch = wm_output_upscaled_padded[
                        :,
                        :,
                        i : i + patch_height + padding_size * 2,
                        j : j + patch_width + padding_size * 2,
                    ]
                    # Run VDSR on the patch
                    us_patch = us_model(patch)
                    # Remove padding from the output patch
                    us_patch = us_patch[
                        :,
                        :,
                        padding_size : -padding_size,
                        padding_size : -padding_size,
                    ]
                    # Place the processed patch back into the output tensor
                    us_output[
                        :,
                        :,
                        i : i + patch_height,
                        j : j + patch_width,
                    ] = us_patch
            # Convert the output tensor back to a PIL image and save
            img = tensor_to_PIL(us_output.squeeze(0).cpu())
            os.makedirs(output_dir, exist_ok=True)
            img.save(out_path)
            logger.info("UPSCALE: processed %s -> %s", inp_path, out_path)
    logger.info("UPSCALE completed in %.3fs", time.perf_counter() - start)
    # Do not track the upscaled directory for cleanup.  We preserve
    # intermediate stages for debugging and reproducibility.
    return output_dir


