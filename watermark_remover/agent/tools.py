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

import requests  # used for downloading images during online scraping

from langchain.agents import tool

# Import transposition helper from utils.  This allows the scraper to
# compute alternate instrument/key suggestions when the requested key
# does not exist in the local library.  It is defined in
# watermark_remover.utils.transposition_utils and copied from the
# original Watermark‑Remover project.
from watermark_remover.utils.transposition_utils import (
    get_transposition_suggestions,
    normalize_key,
)

# Minimal subset of XPaths used for scraping via Selenium.  These values
# originate from the original Watermark Remover project (see
# watermark_remover/download/selenium_utils.py) and are hard‑coded here
# to avoid importing that module.  Only the keys necessary for search
# and basic navigation are included.
# XPaths used by the Selenium scraper.  These values are copied from the
# original Watermark Remover project (see watermark_remover/download/selenium_utils.py).
# Additional keys have been added here to support discovery of song titles,
# artist names, key choices and instrument parts on the PraiseCharts site.
XPATHS = {
    # Search page
    "search_bar": "//*[@id=\"search-input-wrap\"]/input",
    "songs_parent": "//*[@id='page-wrapper']/ion-router-outlet/app-page-search/ion-content/div/div/div/app-search/div",
    # Individual song result fields
    "song_title": "./div/a/div/h5",
    "song_text3": "./div/a/div/span/span",
    "song_text2": "./div/a/div/span",
    "song_image": "./div/div[1]/div/app-product-audio-preview-image/div/img",
    # Template for clicking the nth song in search results; index is 1‑based
    "click_song": "//*[@id='page-wrapper']/ion-router-outlet/app-page-search/ion-content/div/div/div/app-search/div/app-product-list-item[{index}]/div/a/div",
    # Product page buttons and containers
    "chords_button": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[1]/button",
    # Header element that toggles the orchestration section; used to ensure the
    # orchestrations (keys/parts) are loaded.
    "orchestration_header": "//h3[contains(text(), 'Orchestration')]/ancestor::div[4]",
    "key_button": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[3]/app-product-selector-key/div/button",
    "key_parent": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[3]/app-product-selector-key/div/ul",
    "parts_button": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[2]/div/button",
    "parts_parent": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[2]/div/ul",
    "parts_list": "//*[@id='page-wrapper']/ion-router-outlet/app-product-page/ion-content//ul/li/button",
    # Preview image element and next page button
    "image_element": "//*[@id='preview-sheets']/div/div[1]/div/img",
    "next_button": "//button[contains(@class, 'sheet-nav-gradient-button-right')]",
}

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

# If an output directory is mounted (e.g. /app/output) configure the logger
# to also write its events to a file in that directory.  This helps users
# inspect the sequence of steps executed by the tools.  We append to the
# log so successive runs accumulate.  The file is created lazily on
# first import.  Ignore any errors configuring the file handler (for
# example when running in a read‑only environment).
try:
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "pipeline.log"))
    file_handler.setLevel(getattr(logging, _log_level, logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    # Avoid adding duplicate handlers if this module is imported multiple
    # times in the same interpreter session
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "").endswith("pipeline.log")
        for h in logger.handlers
    ):
        logger.addHandler(file_handler)
except Exception:
    # Best‑effort: if we cannot set up file logging, continue silently
    pass


@tool
def scrape_music(
    title: str,
    instrument: str,
    key: str,
    input_dir: str = "data/samples",
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
    # Fast path: if an explicit directory is provided and exists, use it
    if os.path.isdir(input_dir):
        imgs = [
            p
            for p in glob.glob(os.path.join(input_dir, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if imgs:
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
            return input_dir
        # If the directory exists but has no images, fall back to library search
        logger.debug(
            "SCRAPER: explicit directory '%s' exists but contains no images; falling back to library search",
            input_dir,
        )
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

def _scrape_with_selenium(title: str, instrument: str, key: str) -> Optional[str]:
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
    root_dir = os.path.join('data', 'samples', safe_title)
    out_dir = os.path.join(root_dir, norm_key)
    os.makedirs(out_dir, exist_ok=True)

    # Configure headless Chrome
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
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
        # Locate the search bar and enter the title
        search_xpath = XPATHS['search_bar']
        search_el = wait.until(EC.presence_of_element_located((By.XPATH, search_xpath)))
        search_el.clear()
        search_el.send_keys(title)
        # Small pause to allow suggestions/results to render
        time.sleep(2)
        # Gather search results: titles, artists (if available), and index
        song_candidates = []
        try:
            songs_parent = wait.until(EC.presence_of_element_located((By.XPATH, XPATHS['songs_parent'])))
            songs_children = songs_parent.find_elements(By.XPATH, './app-product-list-item')
            for idx, child in enumerate(songs_children, 1):
                # Extract title
                song_title = ''
                artist_name = ''
                try:
                    title_el = child.find_element(By.XPATH, XPATHS['song_title'])
                    song_title = title_el.text.strip()
                except Exception:
                    pass
                # Extract possible artist/instrument fields
                text2 = ''
                text3 = ''
                try:
                    text3_el = child.find_element(By.XPATH, XPATHS['song_text3'])
                    text3 = text3_el.text.strip()
                except Exception:
                    text3 = ''
                if text3:
                    try:
                        text2_el = child.find_element(By.XPATH, XPATHS['song_text2'])
                        text2 = text2_el.text.split("\n")[0].strip()
                    except Exception:
                        text2 = ''
                # Determine artist: in the original GUI, if text3 exists, text2 is the artist name
                if text3 and text2 and text3 != text2:
                    artist_name = text2
                # Append candidate if title exists
                if song_title:
                    song_candidates.append({
                        'index': idx,
                        'title': song_title,
                        'artist': artist_name,
                        'text3': text3,
                    })
        except Exception:
            pass
        # If we found candidates, log them for debugging
        if song_candidates:
            try:
                choices_str = ', '.join([
                    f"{c['title']}" + (f" by {c['artist']}" if c['artist'] else '')
                    for c in song_candidates[:5]
                ])
                logger.info("SCRAPER: search results: %s", choices_str)
            except Exception:
                pass
        # Select the first candidate by default
        if song_candidates:
            selected = song_candidates[0]
            artist_name = selected.get('artist', '')
            song_index = selected['index']
        else:
            # If no candidates, attempt to click the first result anyway
            artist_name = ''
            song_index = 1
        # Click the selected song
        click_xpath = XPATHS['click_song'].format(index=song_index)
        try:
            song_element = wait.until(EC.element_to_be_clickable((By.XPATH, click_xpath)))
            song_element.click()
        except Exception:
            # If clicking fails, abort scraping
            logger.error("SCRAPER: failed to click song result at index %d", song_index)
            return None
        # Optionally click the "Chords & Lyrics" button to ensure the sheet selector panel is visible
        try:
            chords_btn = wait.until(EC.element_to_be_clickable((By.XPATH, XPATHS['chords_button'])))
            chords_btn.click()
        except Exception:
            # Some pages may not have this button or may already be open; ignore errors
            pass
        # Attempt to open the orchestration menu (keys and parts)
        try:
            orch_header = wait.until(EC.element_to_be_clickable((By.XPATH, XPATHS['orchestration_header'])))
            orch_header.click()
        except Exception:
            pass
        # Gather available keys
        available_keys = []
        selected_key = None
        try:
            key_btn = wait.until(EC.element_to_be_clickable((By.XPATH, XPATHS['key_button'])))
            key_btn.click()
            # Wait for the key list to load
            key_parent = wait.until(EC.presence_of_element_located((By.XPATH, XPATHS['key_parent'])))
            key_buttons = key_parent.find_elements(By.TAG_NAME, 'button')
            for btn in key_buttons:
                text = btn.text.strip()
                if text:
                    available_keys.append(text)
            # Determine which key to select: try exact match, else first
            norm_req = normalize_key(key)
            for btn in key_buttons:
                if btn.text.strip().lower() == norm_req.lower():
                    selected_key = btn.text.strip()
                    btn.click()
                    break
            if not selected_key and key_buttons:
                selected_key = key_buttons[0].text.strip()
                key_buttons[0].click()
            # Close key menu (click again)
            try:
                key_btn.click()
            except Exception:
                pass
        except Exception:
            pass
        if available_keys:
            logger.info(
                "SCRAPER: available keys for '%s': %s; selected key: %s",
                title,
                ', '.join(available_keys),
                selected_key or 'none',
            )
        # Gather available instruments
        available_instruments = []
        selected_instrument = None
        try:
            parts_btn = wait.until(EC.element_to_be_clickable((By.XPATH, XPATHS['parts_button'])))
            parts_btn.click()
            parts_parent = wait.until(EC.presence_of_element_located((By.XPATH, XPATHS['parts_parent'])))
            parts_buttons = parts_parent.find_elements(By.TAG_NAME, 'button')
            for btn in parts_buttons:
                part_text = btn.text.strip()
                if not part_text:
                    continue
                # Exclude cover and lead sheet entries as per original logic
                if 'cover' in part_text.lower() or 'lead sheet' in part_text.lower():
                    continue
                available_instruments.append(part_text)
            # Determine which instrument to select: exact match ignoring case
            for btn in parts_buttons:
                part_text = btn.text.strip()
                if not part_text:
                    continue
                if 'cover' in part_text.lower() or 'lead sheet' in part_text.lower():
                    continue
                if part_text.lower() == instrument.lower():
                    selected_instrument = part_text
                    btn.click()
                    break
            if not selected_instrument and available_instruments:
                selected_instrument = available_instruments[0]
                # Click the first valid instrument button
                for btn in parts_buttons:
                    if btn.text.strip() == selected_instrument:
                        btn.click()
                        break
            # Close parts menu
            try:
                parts_btn.click()
            except Exception:
                pass
        except Exception:
            pass
        if available_instruments:
            logger.info(
                "SCRAPER: available instruments for '%s': %s; selected instrument: %s",
                title,
                ', '.join(available_instruments),
                selected_instrument or 'none',
            )
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
    # Ensure that the directory for the PDF exists
    pdf_dir = os.path.dirname(output_pdf) or "."
    os.makedirs(pdf_dir, exist_ok=True)
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