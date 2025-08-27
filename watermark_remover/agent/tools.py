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
import datetime
import shutil
import logging
import os
import re
import time
from typing import Optional, Any
import random

import requests  # used for downloading images during online scraping

# Import the tool decorator from the core tools package.  The decorator used to
# live under ``langchain.agents``, but in recent versions it has moved to
# ``langchain_core.tools``.  We import from the latter to ensure compatibility
# with modern LangChain versions.
from langchain_core.tools import tool

# Import transposition helper from utils.  This allows the scraper to
# compute alternate instrument/key suggestions when the requested key
# does not exist in the local library.  It is defined in
# watermark_remover.utils.transposition_utils and copied from the
# original Watermark‑Remover project.
from watermark_remover.utils.transposition_utils import (
    get_transposition_suggestions,
    normalize_key,
)

# Import Selenium helper and unified XPath definitions from the original project.  These
# definitions mirror those used by the Watermark Remover GUI and ensure that
# our agent navigates PraiseCharts using the same selectors.  See
# watermark_remover/utils/selenium_utils.py for details.
from watermark_remover.utils.selenium_utils import SeleniumHelper, xpaths as XPATHS
from PIL import Image

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

# Global state used by the music scraping and assembly pipeline.  When
# ``scrape_music`` runs, it records metadata about the selected song, artist,
# instrument and key here.  Subsequent steps (e.g. PDF assembly) use this
# metadata to construct meaningful output paths.  ``TEMP_DIRS`` tracks any
# temporary directories created during scraping, watermark removal and
# upscaling so that they can be cleaned up after the final PDF is
# generated.
SCRAPE_METADATA: dict[str, str] = {}
TEMP_DIRS: list[str] = []

# If an output directory is mounted (e.g. /app/output) configure the logger
# to also write its events to a file in that directory.  This helps users
# inspect the sequence of steps executed by the tools.  We append to the
# log so successive runs accumulate.  The file is created lazily on
# first import.  Ignore any errors configuring the file handler (for
# example when running in a read‑only environment).
try:
    # Compute a timestamped log directory under the ``logs`` folder.  The
    # base ``logs`` directory lives at the root of the project (``/app/logs``)
    # and may be bind‑mounted by the user via ``-v $(pwd)/output:/app/logs``.
    # Each run writes its logs and screenshots into a unique timestamped
    # subdirectory.  The WMRA_LOG_DIR environment variable is set to
    # propagate this location to helper modules (e.g. Selenium) so
    # screenshots and other artefacts are saved alongside the logs.
    #
    # Determine a timestamped directory for this run.  Rather than
    # writing logs into a top‑level ``logs`` folder, store them under
    # ``output/logs`` so that all artefacts live inside the output
    # hierarchy.  Each run gets its own unique timestamped subfolder.
    _run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Persist the run timestamp so subsequent tools (e.g. scrape_music)
    # can reuse it when organising their own output.  If another value is
    # already present in the environment then leave it unchanged – this
    # allows callers to override the timestamp (e.g. for testing).  The
    # environment variable is read by scrape_music to locate the run
    # directory for image backups.
    os.environ.setdefault("RUN_TS", _run_ts)
    # Build the log directory under ``output/logs``.  Using ``os.getcwd()``
    # ensures the path is relative to the project root regardless of
    # where the container mounts the code.  Create the directory up front.
    output_dir = os.path.join(os.getcwd(), "output", "logs", os.environ["RUN_TS"])
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WMRA_LOG_DIR"] = output_dir
    # Configure a plain‑text file handler for the pipeline log
    file_handler = logging.FileHandler(os.path.join(output_dir, "pipeline.log"))
    file_handler.setLevel(getattr(logging, _log_level, logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    # Configure a CSV file handler for structured logging.  Each row in
    # pipeline.csv will contain four columns: timestamp, level, logger
    # name and message.  Surround the message in quotes to preserve
    # commas inside the text.  The newline terminator is omitted by
    # default and will be added by the logging module.
    csv_handler = logging.FileHandler(os.path.join(output_dir, "pipeline.csv"))
    csv_handler.setLevel(getattr(logging, _log_level, logging.INFO))
    # Define a custom formatter that adds missing custom attributes with
    # empty strings.  This allows our CSV logs to include columns for
    # button_text, xpath and url even when they are not provided.
    class CsvFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            # Populate optional fields if they are missing
            if not hasattr(record, "button_text"):
                setattr(record, "button_text", "")
            if not hasattr(record, "xpath"):
                setattr(record, "xpath", "")
            if not hasattr(record, "url"):
                setattr(record, "url", "")
            if not hasattr(record, "screenshot"):
                setattr(record, "screenshot", "")
            # Sanitise newlines in all fields to prevent premature row breaks
            for attr in ("button_text", "xpath", "url", "screenshot", "msg", "message"):
                try:
                    val = getattr(record, attr)
                except AttributeError:
                    continue
                if isinstance(val, str):
                    sanitized = val.replace("\n", " ").replace("\r", " ")
                    setattr(record, attr, sanitized)
            return super().format(record)

    csv_formatter = CsvFormatter(
        "%(asctime)s,%(levelname)s,%(name)s,%(button_text)s,%(xpath)s,%(url)s,%(screenshot)s,\"%(message)s\""
    )
    csv_handler.setFormatter(csv_formatter)
    # Avoid adding duplicate handlers if this module is imported
    # multiple times in the same interpreter session.  We identify
    # existing handlers by their output file names.
    existing_files = [
        getattr(h, "baseFilename", "") for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    if not any(f.endswith("pipeline.log") for f in existing_files):
        logger.addHandler(file_handler)
    if not any(f.endswith("pipeline.csv") for f in existing_files):
        logger.addHandler(csv_handler)
except Exception:
    # Best‑effort: if we cannot set up file logging, continue silently
    pass

def sanitize_title(title: str) -> str:
    """Keep A–Z/a–z/0–9, space, (), -, collapse spaces/_ to _, trim _."""
    safe = re.sub(r'[^A-Za-z0-9()\-\s]+', '_', title.strip())
    safe = re.sub(r'[_\s]+', '_', safe)
    return safe.strip('_')

@tool
def scrape_music(
    title: str,
    instrument: str,
    key: str,
    input_dir: Optional[str] = None,
) -> str:
    """Return a directory of sheet music images for the requested title/key.

    This implementation searches a local library under ``data/samples`` for a
    matching piece.  If the exact title and key are not found, it will
    attempt to locate the title in a different key and generate
    transposition suggestions using the helper functions in
    ``watermark_remover.utils.transposition_utils``.  These suggestions
    allow the agent to ask the user to choose an alternate instrument
    and key when the requested key does not exist.

    In summary, the search proceeds as follows:

    * If ``input_dir`` exists and contains images, return it directly.
    * Otherwise, search ``data/samples`` for a subdirectory whose name
      matches the requested ``title`` (case‑insensitive).  This
      directory is expected to contain subdirectories for each key or
      arrangement.
    * Within the matching title directory, look for a subdirectory
      matching ``key`` (case‑insensitive).  If found, return it.
    * If the key is not available, compute alternative instrument/key
      combinations via ``get_transposition_suggestions`` and raise a
      ``ValueError`` with a descriptive message.  The agent can use
      this message to ask the user how to proceed.
    * If no matching title directory is found, raise ``FileNotFoundError``.

    Parameters
    ----------
    title: str
        The title of the piece to search for.
    instrument: str
        The requested instrument (e.g. "piano", "violin").  Used when
        generating transposition suggestions.
    key: str
        The desired concert key (e.g. "C", "Bb", "F#").
    input_dir: str, optional
        Optional explicit path to a directory containing images.  If this
        directory exists, it takes precedence over the library search.

    Returns
    -------
    str
        Path to the directory containing images for the requested title and
        key.

    Raises
    ------
    FileNotFoundError
        If no suitable piece is found in the local library.
    ValueError
        If the title is found but the requested key is missing.  The
        exception message includes suggestions for alternative keys and
        instruments.
    """
    start = time.perf_counter()
    # Normalise key for matching and suggestions
    norm_key = normalize_key(key)
    # Determine a run timestamp and prepare the log hierarchy for this
    # piece.  The run timestamp is read from RUN_TS so all tools use the
    # same folder per request.  If RUN_TS is missing, compute a fresh
    # timestamp.  Sanitize the title to remove symbols and replace
    # whitespace with underscores, mirroring the sanitisation logic used
    # elsewhere in the project.  Backups of the different processing
    # stages will be stored under ``output/logs/<timestamp>/<safe_title>``.
    run_ts = os.environ.get("RUN_TS") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ.setdefault("RUN_TS", run_ts)
    # Sanitise the title for use in directory names.  Replace any
    # characters other than alphanumerics, spaces or hyphens with
    # underscores, then collapse spaces into underscores.
    safe_title = sanitize_title(title)
    # Compute the root of the logs for this run.  Individual subfolders
    # (song/artist/key/instrument) will be created later once we know
    # the artist.  Do not create the per-song directory yet; it will be
    # determined after scraping when the artist is known (if any).
    log_root = os.path.join(os.getcwd(), "output", "logs", run_ts)
    # Sanitize the instrument and key ahead of time.  The key may
    # contain sharps or flats; sanitising it normalises these
    # characters for use in file paths.
    import re
    def _sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    safe_instrument = _sanitize(instrument) if instrument else 'unknown'
    safe_key = _sanitize(key) if key else 'unknown'
    # Record preliminary metadata for later use when assembling the PDF.  We
    # capture the sanitised title, instrument and key along with the run
    # timestamp.  The artist will be filled in after scraping.
    try:
        SCRAPE_METADATA.clear()
        SCRAPE_METADATA.update({
            'title': safe_title,
            'instrument': instrument,
            'key': key,
            'run_ts': run_ts,
        })
    except Exception:
        pass
    # Fast path: if a custom directory (different from the default
    # library root) is provided and exists, copy its images into
    # ``1_original`` and return that path.  This allows callers to
    # override the scraping logic by specifying a specific path.
    root_dir = "data/samples"
    if input_dir and input_dir != root_dir and os.path.isdir(input_dir):
        imgs = [
            p
            for p in glob.glob(os.path.join(input_dir, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if imgs:
            # Determine the artist for logging structure.  When using an
            # explicit directory, no artist information is available,
            # so we default to 'unknown'.
            safe_artist = 'unknown'
            # Build the full instrument directory under the log root:
            # logs/<run_ts>/<song>/<artist>/<key>/<instrument>/
            instrument_dir = os.path.join(
                log_root,
                safe_title,
                safe_artist,
                safe_key,
                safe_instrument,
            )
            original_dir_final = os.path.join(instrument_dir, "1_original")
            os.makedirs(original_dir_final, exist_ok=True)
            # Copy images into the 1_original directory
            for src in imgs:
                try:
                    dst = os.path.join(original_dir_final, os.path.basename(src))
                    shutil.copyfile(src, dst)
                    logger.info("SCRAPER: copied %s -> %s", src, dst)
                except Exception as copy_err:
                    logger.error("SCRAPER: failed to copy %s: %s", src, copy_err)
            # Update metadata with the default artist
            try:
                SCRAPE_METADATA['artist'] = safe_artist
                SCRAPE_METADATA['title'] = safe_title
                SCRAPE_METADATA['instrument'] = instrument
                SCRAPE_METADATA['key'] = key
            except Exception:
                pass
            logger.info(
                "SCRAPER: using explicit directory '%s' (%d image file(s)) for title='%s', instrument='%s', key='%s'",
                input_dir,
                len(imgs),
                title,
                instrument,
                key,
            )
            logger.debug("SCRAPER: sample files: %s", imgs[:5])
            logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
            return original_dir_final
        # If the directory exists but has no images, fall back to scraping
        logger.debug(
            "SCRAPER: explicit directory '%s' exists but contains no images; falling back to scraping",
            input_dir,
        )
    # After checking an explicit directory, perform online scraping.
    # We no longer search a local library.  Instead, attempt to scrape
    # the requested piece.  If scraping fails, compute transposition
    # suggestions and raise an informative error.  This early return
    # ensures that the legacy library search code below is never
    # executed.
    try:
        scraped_dir = _scrape_with_selenium(title, instrument, key)
    except Exception as scrape_err:
        scraped_dir = None
        logger.error("SCRAPER: exception during online scraping: %s", scrape_err)
    if scraped_dir:
        # Determine the artist from metadata (set by _scrape_with_selenium)
        artist_meta = SCRAPE_METADATA.get('artist', '') or 'unknown'
        import re
        def _sanitize(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
        safe_artist = _sanitize(artist_meta)
        # Build the instrument directory under the log root
        instrument_dir = os.path.join(
            log_root,
            safe_title,
            safe_artist,
            safe_key,
            safe_instrument,
        )
        original_dir_final = os.path.join(instrument_dir, "1_original")
        os.makedirs(original_dir_final, exist_ok=True)
        # Copy scraped images into the 1_original directory.
        try:
            imgs = [
                p for p in glob.glob(os.path.join(scraped_dir, "*"))
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
            for src in imgs:
                dst = os.path.join(original_dir_final, os.path.basename(src))
                shutil.copyfile(src, dst)
                logger.info("SCRAPER: copied %s -> %s", src, dst)
        except Exception as copy_err:
            logger.error("SCRAPER: failed to copy scraped files: %s", copy_err)
        # Update metadata with the sanitised artist and title
        try:
            SCRAPE_METADATA['artist'] = artist_meta
            SCRAPE_METADATA['title'] = safe_title
            SCRAPE_METADATA['instrument'] = instrument
            SCRAPE_METADATA['key'] = key
        except Exception:
            pass
        # Remove the temporary scraped directory and its parent as before
        try:
            shutil.rmtree(scraped_dir, ignore_errors=True)
            scrape_parent = os.path.dirname(scraped_dir.rstrip(os.sep))
            if os.path.basename(scrape_parent).startswith(f"{safe_title}_"):
                shutil.rmtree(scrape_parent, ignore_errors=True)
        except Exception:
            pass
        logger.info(
            "SCRAPER: scraped sheet music online for title='%s' instrument='%s' key='%s'",
            title,
            instrument,
            key,
        )
        logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
        return original_dir_final
    # If scraping failed, compute transposition suggestions for the agent to
    # potentially present to the user.  We always raise an error here
    # because there is no local library fallback.
    suggestions = []
    try:
        suggestions = get_transposition_suggestions(instrument, norm_key)
    except Exception:
        suggestions = []
    sugg_str = ", ".join([
        f"{inst}/{k}" for inst, k in suggestions
    ]) if suggestions else "none"
    raise FileNotFoundError(
        f"No matching piece for '{title}' in key '{key}'. Suggestions: {sugg_str}"
    )

    # ------------------------------------------------------------------
    # The code below remains for reference but will never be executed.
    # It performs a legacy search in a local library under data/samples.
    # Since we have removed local library support, this branch is dead.
    # Library root
    root_dir = "data/samples"
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Library directory '{root_dir}' does not exist."
        )
    # Find a directory whose name contains the title (case‑insensitive)
    title_lower = title.lower()
    candidate_dir: Optional[str] = None
    for candidate in sorted(os.listdir(root_dir)):
        cand_path = os.path.join(root_dir, candidate)
        if not os.path.isdir(cand_path):
            continue
        if title_lower in candidate.lower():
            candidate_dir = cand_path
            break
    if candidate_dir is None:
        # Before giving up, attempt to scrape the sheet music online using Selenium.
        try:
            scraped_dir = _scrape_with_selenium(title, instrument, key)
        except Exception as scrape_err:
            # Log any exception that occurs during scraping and fall back
            scraped_dir = None
            logger.error("SCRAPER: exception during online scraping: %s", scrape_err)
        if scraped_dir:
            logger.info(
                "SCRAPER: scraped sheet music online for title='%s' instrument='%s' key='%s' to '%s'",
                title,
                instrument,
                key,
                scraped_dir,
            )
            logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
            return scraped_dir
        # If scraping failed, raise the original file not found error
        raise FileNotFoundError(
            f"No piece matching title '{title}' was found in '{root_dir}', and scraping also failed."
        )
    # Within the candidate directory, look for a subdirectory matching the key
    # We treat each immediate subdirectory as representing a key or arrangement
    available_keys = []
    key_dir: Optional[str] = None
    for sub in sorted(os.listdir(candidate_dir)):
        sub_path = os.path.join(candidate_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        # If the subdirectory contains images, consider it a valid key folder
        images = [
            f
            for f in os.listdir(sub_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if not images:
            continue
        available_keys.append(sub)
        if norm_key.lower() == sub.lower():
            key_dir = sub_path
    if key_dir:
        logger.info(
            "SCRAPER: found local music for title='%s' instrument='%s' key='%s' in '%s'",
            title,
            instrument,
            key,
            key_dir,
        )
        logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
        return key_dir
    # Key not available; compute suggestions
    suggestions = get_transposition_suggestions(available_keys, instrument, norm_key)
    # Build a helpful message
    msg_lines = [
        f"Requested key '{key}' not available for '{title}'.",
        f"Available keys: {', '.join(available_keys) or 'none'}.",
    ]
    direct = suggestions.get('direct') or []
    closest = suggestions.get('closest') or []
    if direct:
        msg_lines.append("Direct transpositions (same concert key) are available:")
        for item in direct:
            msg_lines.append(
                f"  - Instrument: {item['instrument']}, Key: {item['key']}"
            )
    if closest:
        msg_lines.append("Closest alternatives based on minimal transposition:")
        for item in closest:
            msg_lines.append(
                f"  - Instrument: {item['instrument']}, Key: {item['key']} (difference {item['difference']} semitone(s) {item['interval_direction']})"
            )
    message = "\n".join(msg_lines)
    # Log and raise
    logger.warning("SCRAPER: %s", message)
    # Attempt online scraping before raising the error
    try:
        scraped_dir = _scrape_with_selenium(title, instrument, key)
    except Exception as scrape_err:
        scraped_dir = None
        logger.error("SCRAPER: exception during online scraping: %s", scrape_err)
    if scraped_dir:
        logger.info(
            "SCRAPER: scraped sheet music online for title='%s' instrument='%s' key='%s' to '%s'",
            title,
            instrument,
            key,
            scraped_dir,
        )
        logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
        return scraped_dir
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Selenium-based scraper
#
# The following helper function implements dynamic scraping of sheet music
# from praisecharts.com using Selenium.  It is invoked by ``scrape_music``
# when a requested title/key combination cannot be found in the local
# library.  If Selenium or its dependencies are not available, or if
# scraping fails for any reason, the function returns ``None`` so
# ``scrape_music`` can fall back to other logic.

def _scrape_with_selenium(title: str, instrument: str, key: str, *, _retry: bool = False) -> Optional[str]:
    """Attempt to scrape sheet music from an online catalogue.

    This helper uses a headless Chrome browser (via Selenium WebDriver) to
    search for the requested piece on PraiseCharts, navigate to the first
    result, and download the preview sheet images.  Images are saved into
    ``data/samples/<safe_title>/<norm_key>``.  If successful, the path to
    the directory containing the downloaded images is returned.  If any
    errors occur (including missing Selenium dependencies, browser
    start failures, or no images found), ``None`` is returned and an
    error is logged.

    Parameters
    ----------
    title: str
        Piece title to search for.
    instrument: str
        Requested instrument (unused in this implementation but accepted
        for future extensions).
    key: str
        Requested key (used to name the directory).

    Returns
    -------
    Optional[str]
        Path to the directory of downloaded images, or None if scraping failed.
    """
    # Lazy import of Selenium and webdriver_manager.  If either import
    # fails, scraping will be skipped gracefully.
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
        from selenium.webdriver.chrome.service import Service  # type: ignore
        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
        from selenium.webdriver.support import expected_conditions as EC  # type: ignore
        from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    except Exception as e:
        logger.error("SCRAPER: Selenium or webdriver_manager not installed: %s", e)
        return None

    # Sanitize the title to create a safe directory name.  Replace any
    # characters other than alphanumerics, spaces or hyphens with
    # underscores.  Collapse spaces into underscores.
    safe_title = ''.join(
        c if c.isalnum() or c in (' ', '-') else '_' for c in title
    ).strip().replace(' ', '_')
    norm_key = normalize_key(key)
    # Create a temporary directory for this scraping run.  We place
    # temporary directories under the log directory so that they are
    # easy to clean up later.  Use tempfile to avoid collisions.
    import tempfile
    log_root = os.environ.get("WMRA_LOG_DIR", os.getcwd())
    try:
        root_dir = tempfile.mkdtemp(prefix=f"{safe_title}_", dir=log_root)
    except Exception:
        # Fall back to creating under the current working directory
        root_dir = tempfile.mkdtemp(prefix=f"{safe_title}_")
    # Record the temporary directory so it can be removed after the pipeline
    try:
        TEMP_DIRS.append(root_dir)
    except Exception:
        pass
    out_dir = os.path.join(root_dir, norm_key)
    os.makedirs(out_dir, exist_ok=True)

    # Configure headless Chrome
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Set a larger window size to capture more of the page.  A larger viewport
    # (1920x1080) improves screenshot annotations and ensures UI elements
    # appear at predictable coordinates.
    options.add_argument('--window-size=1920,1080')
    # If a system-installed Chromium binary exists, use it; this is
    # necessary when running inside a container where Chrome is installed
    # under /usr/bin.
    for candidate in ('/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome'):
        if os.path.isfile(candidate):
            options.binary_location = candidate
            break
    # Start the WebDriver.  First attempt to use a system‑installed
    # chromedriver (e.g. installed via the ``chromium-driver`` package).
    # Fall back to webdriver_manager only if no local driver is found.
    driver = None
    try:
        driver_path_candidates = [
            '/usr/bin/chromedriver',
            '/usr/lib/chromium-browser/chromedriver',
            '/usr/lib/chromium/chromedriver',
        ]
        for candidate_path in driver_path_candidates:
            if os.path.isfile(candidate_path):
                try:
                    service = Service(candidate_path)
                    driver = webdriver.Chrome(service=service, options=options)
                    break
                except Exception:
                    continue
        if driver is None:
            # Fall back to webdriver_manager: this will download a driver if necessary
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as err:
        logger.error("SCRAPER: failed to start Chrome WebDriver: %s", err)
        return None

    try:
        # Navigate to the search page
        search_url = "https://www.praisecharts.com/search"
        driver.get(search_url)
        wait = WebDriverWait(driver, 10)
        # Use the helper to send keys to the search bar
        if not SeleniumHelper.send_keys_to_element(driver, XPATHS['search_bar'], title, timeout=5, log_func=logger.debug):
            logger.error("SCRAPER: failed to locate or send keys to search bar")
            return None
        # Allow suggestions/results to render
        time.sleep(2)
        # We no longer click the Songs tab here.  Instead, the perform_search
        # helper will handle filtering the results and waiting for the
        # loading spinner to disappear.  This avoids duplicating logic and
        # ensures consistent timing across retries.
        # ------------------------------------------------------------------
        # Helper to perform a search and compile song candidates
        # ------------------------------------------------------------------
        def perform_search() -> list[dict[str, Any]]:
            """Navigate to the search page, enter the title, click the Songs filter and
            return a list of song candidate dictionaries.

            Each dictionary contains the DOM index, song title, artist (if available)
            and text3 (metadata containing keys) for each result.  Albums and
            non-song results are excluded.
            """
            # Navigate to the search URL
            try:
                driver.get(search_url)
            except Exception:
                return []
            # Enter the title in the search bar
            if not SeleniumHelper.send_keys_to_element(driver, XPATHS['search_bar'], title, timeout=5, log_func=logger.debug):
                return []
            # Allow suggestions to render
            time.sleep(random.uniform(0.5, 1.5))
            # Click the Songs tab to filter results.  Use the XPath string directly
            # and ignore errors if the element cannot be clicked.  After clicking,
            # wait for the loading spinner to disappear before collecting
            # search results.  PraiseCharts shows an animated ``app-loading-spinner``
            # element while filtering the results; if we proceed before the
            # spinner disappears, attempts to click the first result may fail.
            try:
                songs_tab_xpath = "//button[contains(., 'Songs')]"
                SeleniumHelper.click_element(driver, songs_tab_xpath, timeout=5, log_func=logger.debug)
                # After clicking the Songs tab, wait for the loading spinner
                # to appear and then disappear.  PraiseCharts renders a
                # <app-loading-spinner> element inside the search bar
                # when filtering; we first wait up to 5 seconds for it to
                # appear, then wait up to an additional 10 seconds for it
                # to vanish.  This prevents premature access to the DOM.
                spinner_xpath = "//app-loading-spinner"
                # Wait for the spinner to appear (if it ever does)
                # Wait up to a few seconds for the spinner to appear.  In some
                # cases the spinner appears after a brief delay.  If it
                # never appears, we will proceed to the disappear wait.
                appear_deadline = time.time() + 7
                while time.time() < appear_deadline:
                    try:
                        if driver.find_elements(By.XPATH, spinner_xpath):
                            break
                    except Exception:
                        pass
                    time.sleep(0.2)
                # Wait for the spinner to disappear.  Some searches can take up to
                # 15 seconds to refresh the results, so allow a generous
                # timeout here.  Poll every 0.3 seconds until no spinner
                # elements are present or until the deadline is reached.
                disappear_deadline = time.time() + 15
                while time.time() < disappear_deadline:
                    try:
                        spinners = driver.find_elements(By.XPATH, spinner_xpath)
                    except Exception:
                        spinners = []
                    if not spinners:
                        break
                    time.sleep(0.3)
                # Brief additional pause to allow DOM to stabilise
                time.sleep(random.uniform(0.5, 1.0))
            except Exception:
                # If clicking the songs tab or waiting for the spinner fails, just continue
                pass
            # Locate the songs container
            songs_parent_el = SeleniumHelper.find_element(driver, XPATHS['songs_parent'], timeout=10, log_func=logger.debug)
            if not songs_parent_el:
                return []
            # Collect children list items
            children = songs_parent_el.find_elements("xpath", './app-product-list-item')
            candidates: list[dict[str, Any]] = []
            for idx, child in enumerate(children, 1):
                song_title = ''
                artist_name = ''
                text3 = ''
                # Extract title
                try:
                    title_el = child.find_element("xpath", XPATHS['song_title'])
                    song_title = title_el.text.strip()
                except Exception:
                    pass
                # Extract text3 (keys or 'Album')
                try:
                    text3_el = child.find_element("xpath", XPATHS['song_text3'])
                    text3 = text3_el.text.strip()
                except Exception:
                    text3 = ''
                # Skip entries without text3 or containing 'album'
                if not text3:
                    continue
                if 'album' in text3.lower():
                    continue
                # Extract artist (text2) if present
                text2 = ''
                try:
                    text2_el = child.find_element("xpath", XPATHS['song_text2'])
                    text2 = text2_el.text.split("\n")[0].strip()
                except Exception:
                    text2 = ''
                if text2 and text3 and text3 != text2:
                    artist_name = text2
                if song_title:
                    candidates.append({
                        'index': idx,
                        'title': song_title,
                        'artist': artist_name,
                        'text3': text3,
                    })
            # Log first few candidates
            if candidates:
                try:
                    choices_str = ', '.join([
                        f"{c['title']}" + (f" by {c['artist']}" if c['artist'] else '')
                        for c in candidates[:5]
                    ])
                    logger.info("SCRAPER: search results: %s", choices_str)
                except Exception:
                    pass
            return candidates

        # Build the initial list of song candidates
        song_candidates = perform_search()
        # Set of attempted song identifiers to avoid infinite loops
        attempted: set[tuple[str, str]] = set()
        selected = None
        artist_name = ''
        song_index = None
        # Loop until a song with orchestration is found or candidates are exhausted
        while True:
            found_candidate = False
            # Iterate through current candidates
            for cand in song_candidates:
                # Unique identifier: (title, artist)
                ident = (cand.get('title', ''), cand.get('artist', ''))
                if ident in attempted:
                    continue
                attempted.add(ident)
                artist_name = cand.get('artist', '') or ''
                song_index = cand['index']
                # Log candidate being evaluated
                try:
                    logger.info(
                        "SCRAPER: evaluating candidate '%s'%s",
                        cand.get('title', 'unknown'),
                        f" by {cand.get('artist')}" if cand.get('artist') else '',
                    )
                except Exception:
                    pass
                # Instead of clicking the candidate in the same tab, open it in a new tab.
                # Retrieve the anchor element for this candidate to extract its href.
                try:
                    candidate_xpath = XPATHS['click_song'].format(index=song_index)
                    candidate_div = SeleniumHelper.find_element(
                        driver, candidate_xpath, timeout=5, log_func=logger.debug
                    )
                    if not candidate_div:
                        raise Exception("candidate_div not found")
                    # The clickable div is inside an <a> tag; get the parent anchor
                    parent_anchor = candidate_div.find_element(By.XPATH, "..")
                    href = parent_anchor.get_attribute("href")
                except Exception as e:
                    logger.error(
                        "SCRAPER: failed to obtain URL for song result at index %d: %s",
                        song_index,
                        e,
                    )
                    # If we cannot extract the song URL we simply skip this candidate.
                    # Do not re‑run the search; continue with the next candidate in the list.
                    continue
                # Save current window handle and open the candidate in a new tab
                original_window = driver.current_window_handle
                try:
                    driver.execute_script("window.open(arguments[0], '_blank');", href)
                except Exception as e:
                    logger.error(
                        "SCRAPER: failed to open new tab for '%s' (%s): %s",
                        cand.get('title', 'unknown'),
                        href,
                        e,
                    )
                    # If opening the new tab fails, skip this candidate and continue.
                    continue
                # Switch to the newly opened tab
                try:
                    driver.switch_to.window(driver.window_handles[-1])
                except Exception:
                    # If switching fails, close the tab and skip this candidate
                    try:
                        driver.close()
                    except Exception:
                        pass
                    try:
                        driver.switch_to.window(original_window)
                    except Exception:
                        pass
                    continue
                # Allow the page to load
                time.sleep(random.uniform(1.0, 2.0))
                # Now operate within the new tab: click chords and orchestration
                try:
                    SeleniumHelper.click_element(
                        driver, XPATHS['chords_button'], timeout=5, log_func=logger.debug
                    )
                    # Pause briefly
                    time.sleep(random.uniform(0.5, 1.0))
                    orch_ok = SeleniumHelper.click_element(
                        driver, XPATHS['orchestration_header'], timeout=5, log_func=logger.debug
                    )
                    time.sleep(random.uniform(0.5, 1.0))
                except Exception:
                    orch_ok = False
                if not orch_ok:
                    # Close the tab and switch back to the original search results tab.
                    try:
                        driver.close()
                    except Exception:
                        pass
                    try:
                        driver.switch_to.window(original_window)
                    except Exception:
                        pass
                    logger.info(
                        "SCRAPER: candidate '%s' has no orchestration; skipping",
                        cand.get('title', 'unknown'),
                    )
                    # Skip to the next candidate without re‑running the search.
                    continue
                else:
                    # Orchestration opened successfully in the new tab
                    selected = cand
                    found_candidate = True
                    break
            # Break outer loop if a candidate with orchestration was found
            if found_candidate:
                break
            # If no candidates left after filtering and we didn't find any, log and abort
            if not song_candidates or all((c.get('title', ''), c.get('artist', '')) in attempted for c in song_candidates):
                logger.error(
                    "SCRAPER: no orchestration found for any search result of '%s'",
                    title,
                )
                # Retry once by restarting search if not yet retried
                if not _retry:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    return _scrape_with_selenium(title, instrument, key, _retry=True)
                return None
        # End of candidate selection loop
        # At this point we are on the product page for the selected song.  We
        # assume the orchestration header was successfully opened.  Proceed to
        # gather keys and instruments.
        available_keys: list[str] = []
        selected_key: str | None = None
        # Open key menu.  If the key menu cannot be opened, skip this candidate
        key_menu_ok = SeleniumHelper.click_element(
            driver, XPATHS['key_button'], timeout=5, log_func=logger.debug
        )
        # Random delay to mimic human interaction after opening the key menu
        time.sleep(random.uniform(0.3, 0.8))
        if not key_menu_ok:
            logger.info(
                "SCRAPER: unable to open key menu for '%s'; skipping candidate",
                selected.get('title', title) if selected else title,
            )
            # Attempt to go back to search results
            try:
                driver.back()
                time.sleep(1)
            except Exception:
                pass
            return None
        # Fetch list items
        key_parent_el = SeleniumHelper.find_element(
            driver, XPATHS['key_parent'], timeout=5, log_func=logger.debug
        )
        if key_parent_el:
            key_buttons = key_parent_el.find_elements(By.TAG_NAME, 'button')
            for btn in key_buttons:
                text = btn.text.strip()
                if text:
                    available_keys.append(text)
            # Choose requested key if available or closest fallback
            requested_norm = normalize_key(key)
            # Determine target semitone if possible
            try:
                from watermark_remover.utils.transposition_utils import KEY_TO_SEMITONE
                target_semitone = KEY_TO_SEMITONE.get(normalize_key(requested_norm), None)
            except Exception:
                target_semitone = None
            # Attempt to select the requested key exactly
            for btn in key_buttons:
                try:
                    btn_text = btn.text.strip()
                except Exception:
                    continue
                if btn_text and btn_text.lower() == requested_norm.lower():
                    selected_key = btn_text
                    try:
                        btn.click()
                    except Exception:
                        pass
                    break
            # If not selected and we know the target semitone, choose the closest key (prefer downward)
            if not selected_key and target_semitone is not None and key_buttons:
                closest = None
                for btn in key_buttons:
                    try:
                        btn_text = btn.text.strip()
                    except Exception:
                        continue
                    if not btn_text:
                        continue
                    # Some key buttons may show enharmonic equivalents separated by '/'
                    name_parts = [normalize_key(part) for part in btn_text.split('/')]
                    # Map to semitones (pick the first valid mapping)
                    semitone_vals = [KEY_TO_SEMITONE.get(p) for p in name_parts if p in KEY_TO_SEMITONE]
                    if not semitone_vals:
                        continue
                    semitone = semitone_vals[0]
                    # Compute signed distance from target key (mod 12)
                    diff = (semitone - target_semitone) % 12
                    if diff > 6:
                        diff -= 12
                    # Candidate tuple: minimize absolute diff, then prefer negative diff (downward)
                    candidate = (abs(diff), 0 if diff < 0 else 1, diff, btn_text, btn)
                    if closest is None or candidate < closest:
                        closest = candidate
                if closest:
                    _, _, _, sel_text, sel_btn = closest
                    selected_key = sel_text
                    try:
                        sel_btn.click()
                    except Exception:
                        pass
            # Fallback: pick first available if no other key selected
            if not selected_key and key_buttons:
                selected_key = key_buttons[0].text.strip()
                try:
                    key_buttons[0].click()
                except Exception:
                    pass
        # Close key menu (click again)
        SeleniumHelper.click_element(
            driver, XPATHS['key_button'], timeout=5, log_func=logger.debug
        )
        # Pause briefly after closing the key menu
        time.sleep(random.uniform(0.3, 0.8))
        if available_keys:
            logger.info(
                "SCRAPER: available keys for '%s': %s; selected key: %s",
                title,
                ', '.join(available_keys),
                selected_key or 'none',
            )
            # If the requested key is not among the available keys, record the
            # fallback choice so the user can understand why a different key
            # was selected.
            try:
                requested_norm = normalize_key(key)
                if selected_key and selected_key.lower() != requested_norm.lower():
                    logger.info(
                        "SCRAPER: requested key '%s' not found; selected fallback key '%s'",
                        requested_norm,
                        selected_key,
                    )
            except Exception:
                pass
        # Gather available instruments
        available_instruments: list[str] = []
        selected_instrument: str | None = None
        # Open parts (instrument) menu.  If it cannot be opened, skip this candidate.
        parts_menu_ok = SeleniumHelper.click_element(
            driver, XPATHS['parts_button'], timeout=5, log_func=logger.debug
        )
        # Random delay to mimic human interaction after opening the parts menu
        time.sleep(random.uniform(0.3, 0.8))
        if not parts_menu_ok:
            logger.info(
                "SCRAPER: unable to open parts menu for '%s'; skipping candidate",
                selected.get('title', title) if selected else title,
            )
            try:
                driver.back()
                time.sleep(1)
            except Exception:
                pass
            return None
        # Locate the parent element for the parts menu and collect buttons
        parts_parent_el = SeleniumHelper.find_element(
            driver, XPATHS['parts_parent'], timeout=5, log_func=logger.debug
        )
        parts_buttons = []
        try:
            from selenium.webdriver.common.by import By  # type: ignore
        except Exception:
            By = None  # type: ignore
        if parts_parent_el is not None and By is not None:
            try:
                parts_buttons = []
                # Scroll to load more instruments; capture new buttons each time
                for _ in range(5):
                    current_buttons = parts_parent_el.find_elements(By.TAG_NAME, 'button')
                    if current_buttons:
                        for b in current_buttons:
                            if b not in parts_buttons:
                                parts_buttons.append(b)
                    # Break if we found a matching instrument already
                    if any(
                        (instrument.lower() in (btn.text or '').lower())
                        or (
                            'horn' in instrument.lower() and 'horn' in (btn.text or '').lower()
                        )
                        for btn in parts_buttons
                    ):
                        break
                    # Scroll the list container
                    try:
                        driver.execute_script(
                            "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight",
                            parts_parent_el,
                        )
                        time.sleep(0.5)
                    except Exception:
                        break
                # Final refresh
                try:
                    current_buttons = parts_parent_el.find_elements(By.TAG_NAME, 'button')
                    for b in current_buttons:
                        if b not in parts_buttons:
                            parts_buttons.append(b)
                except Exception:
                    pass
            except Exception:
                parts_buttons = []
        # Fallback: gather via parts_list if parent not found
        if not parts_buttons:
            parts_buttons = SeleniumHelper.find_elements(
                driver, XPATHS['parts_list'], timeout=5, log_func=logger.debug
            )
        # Populate available instruments excluding covers and lead sheets
        for btn in parts_buttons:
            try:
                part_text = btn.text.strip()
            except Exception:
                part_text = ''
            if not part_text:
                continue
            lower = part_text.lower()
            if 'cover' in lower or 'lead sheet' in lower:
                continue
            if part_text not in available_instruments:
                available_instruments.append(part_text)
        # Determine the most appropriate instrument selection
        requested_lower = instrument.lower()
        # Exact match
        for btn in parts_buttons:
            try:
                part_text = btn.text.strip()
            except Exception:
                continue
            if not part_text:
                continue
            lower = part_text.lower()
            if 'cover' in lower or 'lead sheet' in lower:
                continue
            if lower == requested_lower:
                selected_instrument = part_text
                try:
                    btn.click()
                except Exception:
                    pass
                break
        # Substring/horn match
        if not selected_instrument:
            for btn in parts_buttons:
                try:
                    part_text = btn.text.strip()
                except Exception:
                    continue
                if not part_text:
                    continue
                lower = part_text.lower()
                if 'cover' in lower or 'lead sheet' in lower:
                    continue
                if (
                    requested_lower == lower
                    or requested_lower in lower
                    or lower in requested_lower
                    or (
                        'horn' in requested_lower and 'horn' in lower
                    )
                ):
                    selected_instrument = part_text
                    try:
                        btn.click()
                    except Exception:
                        pass
                    break
        # Default: first available
        if not selected_instrument and available_instruments:
            selected_instrument = available_instruments[0]
            for btn in parts_buttons:
                try:
                    if btn.text.strip() == selected_instrument:
                        btn.click()
                        break
                except Exception:
                    continue
        # Close parts menu
        SeleniumHelper.click_element(
            driver, XPATHS['parts_button'], timeout=5, log_func=logger.debug
        )
        # Pause briefly after closing the parts menu
        time.sleep(random.uniform(0.3, 0.8))
        if available_instruments:
            try:
                logger.info(
                    "SCRAPER: available instruments for '%s': %s; selected instrument: %s",
                    title,
                    ', '.join(available_instruments),
                    selected_instrument or 'none',
                )
                # Log fallback reason
                try:
                    if selected_instrument:
                        # if requested instrument not found or not matching horn rule
                        if not any(
                            selected_instrument.lower() in (instrument or '').lower()
                            or (('horn' in (instrument or '').lower()) and 'horn' in selected_instrument.lower())
                        ):
                            logger.info(
                                "SCRAPER: requested instrument '%s' not found; selected fallback instrument '%s'",
                                instrument,
                                selected_instrument,
                            )
                except Exception:
                    pass
            except Exception:
                pass
        # Now on the product page with the chosen key and instrument.  Locate the image element and next button
        image_xpath = XPATHS['image_element']
        next_button_xpath = XPATHS['next_button']
        downloaded_urls: set[str] = set()
        prev_page_num: Optional[str] = None
        # Loop through preview images.  Use a reasonable upper limit to avoid infinite loops.
        for _ in range(50):
            try:
                image_el = wait.until(EC.presence_of_element_located((By.XPATH, image_xpath)))
            except Exception:
                break
            img_url = image_el.get_attribute('src')
            if not img_url:
                break
            # Extract page number from the filename (e.g. _001.png).  If
            # the page number resets to 001 after the first iteration,
            # assume we've looped through all pages and exit.
            page_num: Optional[str] = None
            try:
                base_name = os.path.basename(img_url)
                if '_' in base_name:
                    part = base_name.split('_')[-1]
                    page_num = part.split('.')[0]
            except Exception:
                page_num = None
            if prev_page_num and page_num == '001':
                break
            # Download if not already retrieved
            if img_url not in downloaded_urls:
                try:
                    resp = requests.get(img_url, timeout=10)
                    if resp.status_code == 200:
                        filename = os.path.basename(img_url)
                        out_path = os.path.join(out_dir, filename)
                        with open(out_path, 'wb') as f:
                            f.write(resp.content)
                        downloaded_urls.add(img_url)
                        logger.info("SCRAPER: downloaded %s", filename)
                except Exception as dl_err:
                    logger.error("SCRAPER: failed to download %s: %s", img_url, dl_err)
            # Navigate to the next page
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
                next_btn.click()
                time.sleep(1)
            except Exception:
                break
            prev_page_num = page_num
        # If we downloaded any images, log artist information if available
        if downloaded_urls:
            if artist_name:
                logger.info("SCRAPER: selected artist: %s", artist_name)
            # Record metadata for use in later pipeline stages.  This metadata
            # includes the sanitised title, selected artist, instrument and key.
            try:
                SCRAPE_METADATA['title'] = safe_title
                SCRAPE_METADATA['artist'] = artist_name or ''
                SCRAPE_METADATA['instrument'] = selected_instrument or ''
                SCRAPE_METADATA['key'] = (selected_key or norm_key) or ''
            except Exception:
                pass
            # Track the downloaded directory for cleanup after the PDF is assembled
            try:
                if out_dir not in TEMP_DIRS:
                    TEMP_DIRS.append(out_dir)
            except Exception:
                pass
            return out_dir
        # Otherwise, scraping failed
        logger.warning("SCRAPER: no images downloaded for title '%s'", title)
        return None
    except Exception as e:
        logger.error("SCRAPER: exception during scraping: %s", e)
        return None
    finally:
        try:
            driver.quit()
        except Exception:
            pass


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


@tool
def upscale_images(input_dir: str, model_dir: str = "models/VDSR", output_dir: str = "upscaled") -> str:
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
        import re
        def _sanitize(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
        title_meta = meta.get('title', '') or 'unknown'
        artist_meta = meta.get('artist', '') or 'unknown'
        key_meta = meta.get('key', '') or 'unknown'
        instrument_meta = meta.get('instrument', '') or 'unknown'
        title_dir = _sanitize(title_meta)
        artist_dir = _sanitize(artist_meta)
        key_dir = _sanitize(key_meta)
        instrument_part = _sanitize(instrument_meta)
        # Build final directory under output/music
        pdf_root = os.path.join(os.getcwd(), "output", "music")
        final_dir = os.path.join(pdf_root, title_dir, artist_dir, key_dir)
        os.makedirs(final_dir, exist_ok=True)
        # Compose file name
        file_title = _sanitize(title_meta.lower()) if title_meta else 'output'
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