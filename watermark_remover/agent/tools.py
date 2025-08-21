"""Tool definitions for the Watermark Remover agents.

Each function in this module is decorated with the ``langchain.agents.tool``
decorator so that it can be invoked by LangChain agents.  The tools provide
an abstraction over the core functionalities of the Watermark Remover
project:

* ``scrape_music``: returns a directory containing sheet music images.  In
  this proof‑of‑concept implementation, no real scraping occurs; the
  function merely returns a pre‑existing directory.  It is left as a stub
  to be replaced with Selenium or API logic as needed.

* ``remove_watermark``: loads a U‑Net model and applies it to each image
  found in the input directory, saving the watermark‑free versions to a new
  directory.

* ``upscale_images``: loads a VDSR model and applies it to each image
  found in the input directory, saving the upscaled results to a new
  directory.

* ``assemble_pdf``: collects all images from a directory and assembles
  them into a multi‑page PDF.

All tools return the path to the directory or file they produce.  If the
specified model directory does not contain weights, the ``load_best_model``
function silently fails and the model runs with randomly initialised weights;
this keeps the example self‑contained and avoids bundling large model
weights in the repository.  Users can populate the ``models/`` directories
with their own trained checkpoints to achieve meaningful results.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import Optional

from langchain.agents import tool

# Import model definitions lazily.  These imports can be heavy and
# require optional dependencies such as torch, torchvision and
# pytorch_msssim.  To avoid import errors when those libraries are not
# installed, we catch ImportError and provide a helpful message at
# runtime.
try:
    import torch  # type: ignore
    from watermark_remover.inference.model_functions import (
        UNet,
        VDSR,
        PIL_to_tensor,
        tensor_to_PIL,
        load_best_model,
    )
except Exception as e:  # broad except to handle ImportError and others
    UNet = VDSR = None  # type: ignore
    PIL_to_tensor = tensor_to_PIL = load_best_model = None  # type: ignore
    torch = None  # type: ignore
    _import_error = e
else:
    _import_error = None

# Set up a module‑level logger.  The log level can be controlled via the
# LOG_LEVEL environment variable (e.g. export LOG_LEVEL=DEBUG).  If not set,
# INFO is used by default.  Only configure the basicConfig once to avoid
# interfering with parent application logging configuration.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    logging.basicConfig(
        level=getattr(logging, _log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
except Exception:
    # basicConfig may have been called elsewhere; ignore errors
    pass
logger = logging.getLogger("wmra.tools")


@tool
def scrape_music(title: str, instrument: str, key: str, input_dir: str = "data/samples") -> str:
    """Return the path to a directory of sheet music images.

    Parameters
    ----------
    title : str
        The title of the piece to search for.  Ignored in this stub implementation.
    instrument : str
        The instrument requested.  Ignored in this stub implementation.
    key : str
        The musical key requested.  Ignored in this stub implementation.
    input_dir : str
        A directory on disk containing images to process.  Defaults to
        ``data/samples`` relative to the project root.

    Returns
    -------
    str
        The same directory specified by ``input_dir``.

    Notes
    -----
    This function is a placeholder for the real scraping logic originally
    implemented in the PyQt GUI with Selenium.  In a production system, you
    could implement network requests or headless browser automation here and
    return a directory of downloaded images.
    """
    start = time.perf_counter()
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    # Count how many images will be processed
    imgs = [p for p in glob.glob(os.path.join(input_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    logger.info(
        "SCRAPER: using directory '%s' (%d image file(s)) for title='%s', instrument='%s', key='%s'",
        input_dir,
        len(imgs),
        title,
        instrument,
        key,
    )
    logger.debug("SCRAPER: sample files: %s", imgs[:5])
    logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
    return input_dir


@tool
def remove_watermark(input_dir: str, model_dir: str = "models/Watermark_Removal", output_dir: str = "processed") -> str:
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
    os.makedirs(output_dir, exist_ok=True)
    # List images to process
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not images:
        raise RuntimeError(f"WMR: no images found in {input_dir}")
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    try:
        load_best_model(model, model_dir)
    except Exception:
        # If model loading fails entirely, continue with random weights.
        logger.warning("WMR: failed to load checkpoints from %s; using random weights", model_dir)
    model.eval()
    count = 0
    for fname in images:
        in_path = os.path.join(input_dir, fname)
        image_tensor = PIL_to_tensor(in_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        out_image = tensor_to_PIL(output.squeeze())
        out_image.save(os.path.join(output_dir, fname))
        count += 1
    elapsed = time.perf_counter() - start
    logger.info(
        "WMR: processed %d image(s) from '%s' -> '%s' on %s in %.3fs",
        count,
        input_dir,
        output_dir,
        device,
        elapsed,
    )
    return output_dir


@tool
def upscale_images(input_dir: str, model_dir: str = "models/VDSR", output_dir: str = "upscaled") -> str:
    """Upscale all images in ``input_dir`` using a VDSR model.

    Parameters
    ----------
    input_dir : str
        Directory containing images to upscale.
    model_dir : str
        Directory containing VDSR checkpoint files.
    output_dir : str
        Directory to which upscaled images are saved.

    Returns
    -------
    str
        Path to the directory containing upscaled images.
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
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not images:
        raise RuntimeError(f"VDSR: no images found in {input_dir}")
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    model = VDSR().to(device)
    try:
        load_best_model(model, model_dir)
    except Exception:
        logger.warning("VDSR: failed to load checkpoints from %s; using random weights", model_dir)
    model.eval()
    count = 0
    for fname in images:
        in_path = os.path.join(input_dir, fname)
        image_tensor = PIL_to_tensor(in_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        out_image = tensor_to_PIL(output.squeeze())
        out_image.save(os.path.join(output_dir, fname))
        count += 1
    elapsed = time.perf_counter() - start
    logger.info(
        "VDSR: processed %d image(s) from '%s' -> '%s' on %s in %.3fs",
        count,
        input_dir,
        output_dir,
        device,
        elapsed,
    )
    return output_dir


@tool
def assemble_pdf(image_dir: str, output_pdf: str = "output.pdf") -> str:
    """Assemble all images in ``image_dir`` into a single PDF.

    Parameters
    ----------
    image_dir : str
        Directory containing images to assemble.
    output_pdf : str
        Filename of the resulting PDF.  If the directory part is omitted,
        the file will be created in the current working directory.

    Returns
    -------
    str
        Path to the created PDF.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    try:
        from reportlab.pdfgen import canvas  # type: ignore
        from reportlab.lib.pagesizes import letter  # type: ignore
    except ImportError as e:
        raise ImportError(
            "reportlab is required for PDF assembly. Please install it via\n"
            "`pip install reportlab`."
        ) from e
    start = time.perf_counter()
    images = [
        f
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    output_path = os.path.abspath(output_pdf)
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    for img in images:
        img_path = os.path.join(image_dir, img)
        c.drawImage(img_path, 0, 0, width=width, height=height, preserveAspectRatio=True, anchor='c')
        c.showPage()
    c.save()
    elapsed = time.perf_counter() - start
    logger.info(
        "ASSEMBLER: assembled %d image(s) from '%s' into PDF '%s' in %.3fs",
        len(images),
        image_dir,
        output_path,
        elapsed,
    )
    return output_path