"""PDF assembly tool for the Watermark Remover pipeline."""

from __future__ import annotations

import os
import shutil
import time

from langchain_core.tools import tool

from .scrape import SCRAPE_METADATA, sanitize_title, logger


@tool
def assemble_pdf(image_dir: str, output_pdf: str = "output/output.pdf") -> str:
    """Assemble images from a directory into a single PDF file.

    Parameters
    ----------
    image_dir : str
        Directory containing images to assemble into a PDF.
    output_pdf : str
        Name or path of the output PDF file.  If not provided, defaults to
        ``"output/output.pdf"`` so that PDFs are written to the ``output``
        directory relative to the current working directory.  If a
        directory component is included in ``output_pdf``, it will be
        created automatically.

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
    images = [
        f
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]
    if not images:
        raise RuntimeError(f"PDF: no images found in {image_dir}")
    # Determine the output paths for the final PDF.  We produce two
    # copies: one stored under ``output/music/<title>/<artist>/<key>/``
    # organised by song, artist and key, and another under the log
    # directory ``output/logs/<run_ts>/<title>/4_final_pdf`` for
    # debugging.  Use metadata recorded during scraping to build a
    # meaningful directory structure.  When metadata is missing, fall
    # back to the provided ``output_pdf`` parameter.
    final_pdf_path: str
    debug_pdf_path: str
    try:
        meta = SCRAPE_METADATA.copy()
        # Retrieve run timestamp for logs
        run_ts = meta.get('run_ts') or os.environ.get('RUN_TS') or ''
        # Use sanitised components for file and directory names.  The
        # title stored in metadata is already sanitised.  If any
        # component is missing, substitute a sensible default.
        # Use unified sanitiser for title, artist, key and instrument names
        title_meta = meta.get('title', '') or 'unknown'
        artist_meta = meta.get('artist', '') or 'unknown'
        key_meta = meta.get('key', '') or 'unknown'
        instrument_meta = meta.get('instrument', '') or 'unknown'
        title_dir = sanitize_title(title_meta)
        artist_dir = sanitize_title(artist_meta)
        key_dir = sanitize_title(key_meta)
        instrument_part = sanitize_title(instrument_meta)
        # Build final directory under output/music
        pdf_root = os.path.join(os.getcwd(), "output", "music")
        final_dir = os.path.join(pdf_root, title_dir, artist_dir, key_dir)
        os.makedirs(final_dir, exist_ok=True)
        # Compose file name
        file_title = sanitize_title(title_meta.lower()) if title_meta else 'output'
        file_name = f"{file_title}_{instrument_part}_{key_dir}.pdf"
        final_pdf_path = os.path.join(final_dir, file_name)
        # Build debug directory under logs mirroring the song/artist/key/instrument
        # structure.  If a run timestamp is available, construct the full
        # hierarchy; otherwise fall back to a directory adjacent to the
        # input images.  The instrument component comes from instrument_part.
        if run_ts:
            debug_dir = os.path.join(
                os.getcwd(),
                "output",
                "logs",
                run_ts,
                title_dir,
                artist_dir,
                key_dir,
                instrument_part,
                "4_final_pdf",
            )
        else:
            # Fallback: use the parent of the input image directory
            debug_dir = os.path.join(os.path.dirname(image_dir.rstrip(os.sep)), "4_final_pdf")
        os.makedirs(debug_dir, exist_ok=True)
        debug_pdf_path = os.path.join(debug_dir, file_name)
    except Exception:
        # Fallback: use provided output_pdf and create directories if necessary
        final_pdf_path = output_pdf
        debug_pdf_path = output_pdf
        os.makedirs(os.path.dirname(final_pdf_path) or '.', exist_ok=True)
    # Prepare the canvas for the PDF.  Render pages to the final
    # destination first, then copy the resulting file into the debug
    # directory.  This avoids rendering twice.
    c = canvas.Canvas(final_pdf_path, pagesize=letter)
    width, height = letter
    for fname in images:
        path = os.path.join(image_dir, fname)
        c.drawImage(path, 0, 0, width=width, height=height, preserveAspectRatio=True)
        c.showPage()
    c.save()
    # Copy the final PDF into the debug directory for debugging purposes
    try:
        if final_pdf_path != debug_pdf_path:
            shutil.copyfile(final_pdf_path, debug_pdf_path)
    except Exception as cp_err:
        logger.error("ASSEMBLER: failed to copy PDF to debug directory: %s", cp_err)
    logger.info("ASSEMBLER: wrote %d pages to %s", len(images), final_pdf_path)
    logger.info("ASSEMBLER completed in %.3fs", time.perf_counter() - start)
    # Do not clean up intermediate directories.  All stages are
    # preserved under the run directory for future debugging and
    # reproducibility.  Return the path to the assembled PDF.
    return final_pdf_path
