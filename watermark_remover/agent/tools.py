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
def scrape_music(
    title: str,
    instrument: str,
    key: str,
    input_dir: str = "data/samples",
) -> str:
    """Return the path to a directory of sheet music images.

    This stub implementation simply returns a directory on disk.  If the
    specified ``input_dir`` exists, it is returned directly.  If it does not
    exist, the function attempts to locate a fallback directory under
    ``data/samples`` that contains image files.  The first such directory
    discovered is used.  If no suitable fallback is found, a FileNotFoundError
    is raised.

    Parameters
    ----------
    title : str
        The title of the piece to search for.  Used to help locate
        alternative directories if the specified ``input_dir`` does not
        exist.  Matching is case‑insensitive and partial.
    instrument : str
        The instrument requested.  Unused in this stub but provided for
        future expansion.
    key : str
        The musical key requested.  Unused in this stub but provided for
        future expansion.
    input_dir : str, optional
        A directory on disk containing images to process.  Defaults to
        ``"data/samples"`` relative to the project root.

    Returns
    -------
    str
        Path to the directory containing images.  This may be the
        ``input_dir`` provided or a fallback under ``data/samples``.

    Raises
    ------
    FileNotFoundError
        If neither the specified ``input_dir`` nor any fallback directory
        exists.

    Notes
    -----
    This function acts as a bridge between the LLM‑driven agent and the
    underlying storage.  In a production system you would replace this
    implementation with actual scraping logic (e.g. Selenium or API calls)
    that downloads sheet music based on the provided ``title``,
    ``instrument`` and ``key``.
    """
    start = time.perf_counter()
    # If the requested directory exists, use it directly.
    if os.path.isdir(input_dir):
        imgs = [
            p
            for p in glob.glob(os.path.join(input_dir, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
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
    # Requested directory does not exist.  Search under data/samples for a
    # fallback directory.  Use the provided title to try to find a match.
    # We search for directories containing image files and whose name
    # contains the title (case‑insensitive).  If none match, we select
    # the first directory with any images.  If data/samples does not
    # exist or no directories contain images, we raise FileNotFoundError.
    root_dir = "data/samples"
    fallback: Optional[str] = None
    title_lower = (title or "").lower()
    if os.path.isdir(root_dir):
        # Walk immediate subdirectories of root_dir (non‑recursive) to find
        # candidate directories.  This avoids descending into deeply nested
        # structures and keeps fallback selection simple.
        for candidate in sorted(os.listdir(root_dir)):
            cand_path = os.path.join(root_dir, candidate)
            if not os.path.isdir(cand_path):
                continue
            # Gather image files in this candidate directory
            images = [
                f
                for f in os.listdir(cand_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
            if not images:
                continue
            # If the directory name contains the title string, prefer it.
            if title_lower and title_lower in candidate.lower():
                fallback = cand_path
                break
            # Otherwise record it as a potential fallback if we haven't
            # found a title match yet.
            if fallback is None:
                fallback = cand_path
        if fallback:
            logger.warning(
                "SCRAPER: directory '%s' not found; falling back to '%s'",
                input_dir,
                fallback,
            )
            logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
            return fallback
    # No fallback found; raise an error.
    raise FileNotFoundError(
        f"Input directory {input_dir} does not exist and no fallback directories were found in '{root_dir}'."
    )


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
            tensor = PIL_to_tensor(inp_path)
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            img = tensor_to_PIL(output.squeeze(0).cpu())
        os.makedirs(processed_dir, exist_ok=True)
        img.save(out_path)
        logger.info("WMR: processed %s -> %s", inp_path, out_path)
    logger.info("WMR completed in %.3fs", time.perf_counter() - start)
    return processed_dir


@tool
def upscale_images(input_dir: str, model_dir: str = "models/VDSR", output_dir: str = "upscaled") -> str:
    """Upscale all images in ``input_dir`` using a VDSR model.

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
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not images:
        raise RuntimeError(f"UPSCALE: no images found in {input_dir}")
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    model = VDSR().to(device)
    # Load the best checkpoint if available.  If loading fails the model
    # will continue with random weights so the pipeline still runs.
    try:
        load_best_model(model, model_dir)  # type: ignore[misc]
    except Exception:
        logger.warning("VDSR: failed to load checkpoints from %s; using random weights", model_dir)
    model.eval()
    processed_dir = output_dir
    for fname in images:
        inp_path = os.path.join(input_dir, fname)
        out_path = os.path.join(processed_dir, fname)
        with torch.no_grad():
            tensor = PIL_to_tensor(inp_path)
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            img = tensor_to_PIL(output.squeeze(0).cpu())
        os.makedirs(processed_dir, exist_ok=True)
        img.save(out_path)
        logger.info("UPSCALE: processed %s -> %s", inp_path, out_path)
    logger.info("UPSCALE completed in %.3fs", time.perf_counter() - start)
    return processed_dir


@tool
def assemble_pdf(image_dir: str, output_pdf: str = "output.pdf") -> str:
    """Assemble images from a directory into a single PDF file.

    Parameters
    ----------
    image_dir : str
        Directory containing images to assemble into a PDF.
    output_pdf : str
        Name of the output PDF file.  If not provided, defaults to
        ``"output.pdf"``.

    Returns
    -------
    str
        Path to the created PDF file.

    Notes
    -----
    If reportlab cannot be imported, this function will raise an
    ImportError.  The PDF will contain one page per image, preserving
    the original aspect ratio.
    """
    start = time.perf_counter()
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise ImportError(f"reportlab is required for PDF assembly: {e}")
    images = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not images:
        raise RuntimeError(f"PDF: no images found in {image_dir}")
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    for fname in images:
        path = os.path.join(image_dir, fname)
        c.drawImage(path, 0, 0, width=width, height=height, preserveAspectRatio=True)
        c.showPage()
    c.save()
    logger.info("ASSEMBLER: wrote %d pages to %s", len(images), output_pdf)
    logger.info("ASSEMBLER completed in %.3fs", time.perf_counter() - start)
    return output_pdf