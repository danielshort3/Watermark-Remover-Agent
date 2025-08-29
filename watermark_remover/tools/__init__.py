"""Convenience imports for Watermark Remover tools."""

from .scrape import (
    scrape_music,
    sanitize_title,
    SCRAPE_METADATA,
    TEMP_DIRS,
    logger,
)
from .watermark import remove_watermark
from .upscale import upscale_images
from .pdf import assemble_pdf

__all__ = [
    "scrape_music",
    "remove_watermark",
    "upscale_images",
    "assemble_pdf",
    "sanitize_title",
    "SCRAPE_METADATA",
    "TEMP_DIRS",
    "logger",
]

