"""Tool definitions for the Watermark Remover agents.

This module defines the tool functions used by the Watermark‑Remover
agent.  The ``scrape_music`` tool now relies exclusively on the
Selenium‑based scraper and no longer falls back to the ``data/samples``
directory.  All logs and screenshots are written into a timestamped
subdirectory under ``logs/``.  When the scraper must select a key
or instrument that differs from the user's request, it records a
reasoning message in the log.  If a selected song has no
orchestration available, the scraper will return to the search
results and evaluate the next candidate rather than looping.
"""

from __future__ import annotations

import datetime
import glob
import logging
import os
import shutil
import time
from typing import Any, Optional

import requests  # used for downloading images during online scraping

from langchain_core.tools import tool

from watermark_remover.utils.transposition_utils import (
    get_transposition_suggestions,
    normalize_key,
)
from watermark_remover.utils.selenium_utils import SeleniumHelper, xpaths as XPATHS

# Lazy import of heavy model dependencies.  If torch is not installed
try:
    import torch  # type: ignore
    from watermark_remover.inference.model_functions import (
        UNet,
        VDSR,
        load_best_model,
    )
except Exception:
    torch = None  # type: ignore
    UNet = VDSR = load_best_model = None  # type: ignore

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -----------------------------------------------------------------------------
# Logging configuration
#
# Logs are written into a timestamped subdirectory under ``logs`` in the
# project root.  The directory is created at import time.  Two handlers
# write plain text and CSV records of each action, and the Selenium
# helper writes screenshots into the same directory.

# Initialise a timestamped log directory
#
# Logs should live under ``/app/logs`` when running in the Docker
# container.  The user will typically mount a host directory onto
# ``/app/logs`` using ``-v $(pwd)/output:/app/logs``.  To respect
# this mount, use the absolute ``/app/logs`` path rather than
# ``os.getcwd()/logs``.  A fallback to the current working
# directory is retained for non‑container usage.  The timestamp
# subdirectory isolates each run and is exported via
# ``WMRA_LOG_DIR`` so that other modules (e.g. graph_ollama) know
# where to write thought logs.
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_base = os.environ.get("WMRA_LOG_PARENT", "/app/logs")
if not os.path.isabs(_log_base):
    # If a relative path is provided, make it relative to the CWD
    _log_base = os.path.join(os.getcwd(), _log_base)
_log_root = os.path.join(_log_base, _timestamp)
os.makedirs(_log_root, exist_ok=True)
os.environ["WMRA_LOG_DIR"] = _log_root

logger = logging.getLogger("wmra.tools")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

def _setup_logging() -> None:
    """Configure logging handlers and formatters.

    A plain text handler and a CSV handler are attached to the module
    logger.  The CSV log includes separate columns for timestamp,
    level, logger name, button text, XPath, URL, screenshot path and
    message.  Any newline characters in the log message are replaced
    with spaces to preserve CSV row integrity.
    """
    if getattr(logger, "_configured", False):
        return
    # Plain text handler
    text_path = os.path.join(_log_root, "pipeline.log")
    text_handler = logging.FileHandler(text_path, encoding="utf-8")
    text_format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    text_handler.setFormatter(text_format)
    logger.addHandler(text_handler)
    # CSV handler
    csv_path = os.path.join(_log_root, "pipeline.csv")
    csv_handler = logging.FileHandler(csv_path, encoding="utf-8")
    class CsvFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            # Extract structured extras if present
            btn = getattr(record, "button_text", "") or ""
            xpath = getattr(record, "xpath", "") or ""
            url = getattr(record, "url", "") or ""
            screenshot = getattr(record, "screenshot", "") or ""
            msg = record.getMessage().replace("\n", " ").replace("\r", " ")
            return f"{self.formatTime(record)};{record.levelname};{record.name};{btn};{xpath};{url};{screenshot};{msg}"
    csv_format = CsvFormatter()
    csv_handler.setFormatter(csv_format)
    logger.addHandler(csv_handler)
    logger._configured = True  # type: ignore[attr-defined]

_setup_logging()

# -----------------------------------------------------------------------------
# Global variables for metadata and cleanup

SCRAPE_METADATA: dict[str, str] = {}
TEMP_DIRS: list[str] = []

# -----------------------------------------------------------------------------
# Utility functions

def _cleanup_temp_dirs() -> None:
    """Remove temporary directories created during processing."""
    for d in list(TEMP_DIRS):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    TEMP_DIRS.clear()

def _record_meta(**kwargs: str) -> None:
    """Update scrape metadata with provided key/value pairs."""
    SCRAPE_METADATA.update({k: v for k, v in kwargs.items() if v})

# -----------------------------------------------------------------------------
# Tool definitions

@tool
def scrape_music(*, title: str, instrument: str, key: str, input_dir: Optional[str] = None) -> Optional[str]:
    """
    Locate and download sheet music for the given song.  This tool uses
    Selenium to search PraiseCharts for ``title`` and attempts to
    retrieve preview images for the requested ``instrument`` and
    ``key``.  It no longer falls back to the ``data/samples`` folder.

    If the requested key or instrument is unavailable, the scraper
    selects a fallback and records its reasoning in the log.  If the
    current candidate has no orchestration (i.e., the key or parts
    menus cannot be opened), the scraper moves to the next search
    result.  On success, returns a directory containing downloaded
    image files.  Returns ``None`` when no suitable candidate is found
    or an unrecoverable error occurs.
    """
    start = time.perf_counter()
    # Only honour input_dir if explicitly provided by the caller.
    if input_dir:
        return input_dir
    # Perform the online scrape
    out_dir = _scrape_with_selenium(title, instrument, key)
    if out_dir:
        logger.info("SCRAPER completed in %.3fs", time.perf_counter() - start)
    else:
        logger.error("SCRAPER failed for title='%s' instrument='%s' key='%s'", title, instrument, key)
    return out_dir


def _scrape_with_selenium(title: str, instrument: str, key: str) -> Optional[str]:
    """Perform the Selenium scraping routine.

    Searches PraiseCharts for the given ``title`` and iterates through the
    search results.  For each candidate, it attempts to open the
    orchestration menu, choose the requested key and instrument (or a
    fallback) and download preview images.  If a candidate lacks
    orchestration (i.e., cannot open the key or parts menus), the
    scraper returns to the search page and tries the next candidate.
    Returns the directory containing downloaded images on success or
    None on failure.
    """
    # Prepare a temporary output directory for this scrape
    out_dir = os.path.join(os.getcwd(), f"tmp_scrape_{int(time.time() * 1000)}")
    os.makedirs(out_dir, exist_ok=True)
    TEMP_DIRS.append(out_dir)
    driver = None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        options = webdriver.ChromeOptions()
        # Launch headless Chrome with a fixed window size
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        wait = WebDriverWait(driver, 10)
        # Navigate to search page and enter the query
        search_url = "https://www.praisecharts.com/search"
        driver.get(search_url)
        # Small random pause to allow dynamic content to load
        import random
        time.sleep(random.uniform(0.5, 1.5))
        if not SeleniumHelper.send_keys_to_element(driver, XPATHS['search_bar'], title, timeout=5, log_func=logger.debug):
            logger.error("SCRAPER: failed to locate or send keys to search bar")
            return None
        # Give results time to load
        time.sleep(random.uniform(1.0, 2.0))
        songs_parent = SeleniumHelper.find_element(driver, XPATHS['songs_parent'], timeout=10, log_func=logger.debug)
        if not songs_parent:
            logger.error("SCRAPER: no search results found for '%s'", title)
            return None
        songs_children = songs_parent.find_elements("xpath", './app-product-list-item')
        # Compile candidate list with indices, titles and artists
        candidates: list[dict[str, Any]] = []
        for idx, child in enumerate(songs_children, 1):
            song_title = ''
            artist_name = ''
            try:
                song_title = child.find_element("xpath", XPATHS['song_title']).text.strip()
            except Exception:
                pass
            text3 = ''
            try:
                text3 = child.find_element("xpath", XPATHS['song_text3']).text.strip()
            except Exception:
                pass
            text2 = ''
            if text3:
                try:
                    text2 = child.find_element("xpath", XPATHS['song_text2']).text.split("\n")[0].strip()
                except Exception:
                    text2 = ''
            if text3 and text2 and text3 != text2:
                artist_name = text2
            if song_title:
                candidates.append({'index': idx, 'title': song_title, 'artist': artist_name})
        # Iterate through candidates until one yields orchestration
        for cand in candidates:
            cand_title = cand['title']
            cand_artist = cand['artist']
            logger.info("SCRAPER: evaluating candidate '%s' by %s", cand_title, cand_artist or 'unknown')
            # Click the song entry
            if not SeleniumHelper.click_dynamic_element(driver, XPATHS['click_song'], cand['index'], timeout=5, log_func=logger.debug):
                logger.error("SCRAPER: failed to click song result at index %d", cand['index'])
                continue
            # Brief pause after clicking a result to allow the page to load
            time.sleep(random.uniform(0.7, 1.5))
            # Attempt to open chords & lyrics and orchestration
            SeleniumHelper.click_element(driver, XPATHS['chords_button'], timeout=5, log_func=logger.debug)
            # Random wait to mimic human behavior before clicking orchestration
            time.sleep(random.uniform(0.5, 1.2))
            if not SeleniumHelper.click_element(driver, XPATHS['orchestration_header'], timeout=5, log_func=logger.debug):
                logger.info("SCRAPER: candidate '%s' has no orchestration; skipping", cand_title)
                driver.back()
                # Wait briefly before moving to next candidate
                time.sleep(random.uniform(0.5, 1.2))
                continue
            # Keys
            available_keys: list[str] = []
            selected_key: Optional[str] = None
            if SeleniumHelper.click_element(driver, XPATHS['key_button'], timeout=5, log_func=logger.debug):
                key_parent_el = SeleniumHelper.find_element(driver, XPATHS['key_parent'], timeout=5, log_func=logger.debug)
                if key_parent_el:
                    key_buttons = key_parent_el.find_elements(By.TAG_NAME, 'button')
                    for btn in key_buttons:
                        txt = btn.text.strip()
                        if txt:
                            available_keys.append(txt)
                    norm_req = normalize_key(key)
                    for btn in key_buttons:
                        if btn.text.strip().lower() == norm_req.lower():
                            selected_key = btn.text.strip()
                            btn.click()
                            break
                    if not selected_key and key_buttons:
                        selected_key = key_buttons[0].text.strip()
                        key_buttons[0].click()
                    # Log fallback reasoning
                    if available_keys and selected_key and selected_key.lower() != normalize_key(key).lower():
                        logger.info(
                            "SCRAPER: requested key '%s' not available; using fallback '%s' from %s",
                            key, selected_key, ', '.join(available_keys),
                        )
                # Collapse the key menu; pause afterwards
                SeleniumHelper.click_element(driver, XPATHS['key_button'], timeout=5, log_func=logger.debug)
                time.sleep(random.uniform(0.3, 0.8))
            # Instruments
            available_instruments: list[str] = []
            selected_instrument: Optional[str] = None
            if SeleniumHelper.click_element(driver, XPATHS['parts_button'], timeout=5, log_func=logger.debug):
                parts_parent_el = SeleniumHelper.find_element(driver, XPATHS['parts_parent'], timeout=5, log_func=logger.debug)
                parts_buttons: list = []
                if parts_parent_el:
                    # Scroll to bottom to load all instruments
                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", parts_parent_el)
                    parts_buttons = parts_parent_el.find_elements(By.TAG_NAME, 'button')
                else:
                    # Fallback: get all buttons globally
                    parts_buttons = driver.find_elements("xpath", XPATHS['parts_list'])
                for btn in parts_buttons:
                    try:
                        txt = btn.text.strip()
                    except Exception:
                        continue
                    if not txt:
                        continue
                    lower = txt.lower()
                    if 'cover' in lower or 'lead sheet' in lower:
                        continue
                    available_instruments.append(txt)
                req_lower = instrument.lower()
                # Exact match
                for btn in parts_buttons:
                    try:
                        txt = btn.text.strip()
                    except Exception:
                        continue
                    if not txt:
                        continue
                    if txt.lower() == req_lower:
                        selected_instrument = txt
                        try:
                            btn.click()
                        except Exception:
                            pass
                        break
                # Substring match (horn synonyms)
                if not selected_instrument:
                    for btn in parts_buttons:
                        try:
                            txt = btn.text.strip()
                        except Exception:
                            continue
                        lower = txt.lower()
                        if req_lower in lower or lower in req_lower or ('horn' in req_lower and 'horn' in lower):
                            selected_instrument = txt
                            try:
                                btn.click()
                            except Exception:
                                pass
                            break
                # Fallback to first instrument
                if not selected_instrument and available_instruments:
                    selected_instrument = available_instruments[0]
                    for btn in parts_buttons:
                        try:
                            if btn.text.strip() == selected_instrument:
                                btn.click()
                                break
                        except Exception:
                            continue
                # Log reasoning for fallback instrument
                if available_instruments and selected_instrument and selected_instrument.lower() != req_lower:
                    logger.info(
                        "SCRAPER: requested instrument '%s' not available; using fallback '%s' from %s",
                        instrument, selected_instrument, ', '.join(available_instruments),
                    )
                # Collapse the parts menu and pause
                SeleniumHelper.click_element(driver, XPATHS['parts_button'], timeout=5, log_func=logger.debug)
                time.sleep(random.uniform(0.3, 0.8))
            # If we have at least one key and instrument, download images
            if selected_key and selected_instrument:
                # Save metadata
                _record_meta(title=cand_title, artist=cand_artist or '', key=selected_key, instrument=selected_instrument)
                # Download preview images
                image_xpath = XPATHS['image_element']
                next_button_xpath = XPATHS['next_button']
                downloaded: set[str] = set()
                prev_page = None
                for _ in range(50):
                    try:
                        image_el = wait.until(EC.presence_of_element_located((By.XPATH, image_xpath)))
                    except Exception:
                        break
                    img_url = image_el.get_attribute('src')
                    if not img_url:
                        break
                    page_num = None
                    try:
                        part = os.path.basename(img_url).split('_')[-1]
                        page_num = part.split('.')[0]
                    except Exception:
                        page_num = None
                    if prev_page and page_num == '001':
                        break
                    if img_url not in downloaded:
                        try:
                            resp = requests.get(img_url, timeout=10)
                            if resp.status_code == 200:
                                fname = os.path.basename(img_url)
                                with open(os.path.join(out_dir, fname), 'wb') as f:
                                    f.write(resp.content)
                                downloaded.add(img_url)
                                logger.info("SCRAPER: downloaded %s", fname)
                        except Exception as dl_err:
                            logger.error("SCRAPER: failed to download %s: %s", img_url, dl_err)
                    # Next page
                    try:
                        next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
                        next_btn.click()
                        time.sleep(1)
                    except Exception:
                        break
                    prev_page = page_num
                # If downloaded, return directory
                if downloaded:
                    if cand_artist:
                        logger.info("SCRAPER: selected artist: %s", cand_artist)
                    return out_dir
            # Not successful: go back and try next candidate
            driver.back()
        # No candidate succeeded
        return None
    except Exception as exc:
        logger.error("SCRAPER: unhandled exception: %s", exc)
        return None
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


@tool
def remove_watermark(*, input_dir: str, model_dir: Optional[str] = None) -> Optional[str]:
    """Remove watermarks from each image in the input directory."""
    start = time.perf_counter()
    # Early exit if torch is unavailable
    if torch is None or UNet is None:
        logger.warning("WMR: torch or UNet unavailable; skipping watermark removal")
        return input_dir
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        if model_dir and os.path.isdir(model_dir):
            load_best_model(model, os.path.join(model_dir, 'model'))
        model.eval()
        # Create output directory
        out_dir = os.path.join(os.getcwd(), f"wmr_{int(time.time() * 1000)}")
        os.makedirs(out_dir, exist_ok=True)
        TEMP_DIRS.append(out_dir)
        for img_path in glob.glob(os.path.join(input_dir, '*.png')):
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img_tensor = torch.tensor((list(img.getdata())), dtype=torch.float32).reshape((img.height, img.width, 3)).permute(2,0,1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    out = model(img_tensor.to(device)).cpu().squeeze(0)
                out_img = Image.fromarray((out.permute(1,2,0).numpy() * 255).astype('uint8'))
                out_path = os.path.join(out_dir, os.path.basename(img_path))
                out_img.save(out_path)
                logger.info("WMR: processed %s -> %s", img_path, out_path)
            except Exception as e:
                logger.error("WMR: failed to process %s: %s", img_path, e)
        logger.info("WMR completed in %.3fs", time.perf_counter() - start)
        return out_dir
    except Exception as exc:
        logger.error("WMR: error during watermark removal: %s", exc)
        return input_dir


@tool
def upscale_images(*, input_dir: str, model_dir: Optional[str] = None) -> Optional[str]:
    """Upscale images in the input directory using a VDSR model."""
    start = time.perf_counter()
    if torch is None or VDSR is None:
        logger.warning("UPSCALE: torch or VDSR unavailable; skipping upscaling")
        return input_dir
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VDSR().to(device)
        if model_dir and os.path.isdir(model_dir):
            load_best_model(model, os.path.join(model_dir, 'model'))
        model.eval()
        out_dir = os.path.join(os.getcwd(), f"upscl_{int(time.time() * 1000)}")
        os.makedirs(out_dir, exist_ok=True)
        TEMP_DIRS.append(out_dir)
        for img_path in glob.glob(os.path.join(input_dir, '*.png')):
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img).astype('float32') / 255.0
                img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0)
                with torch.no_grad():
                    out = model(img_tensor.to(device)).cpu().squeeze(0)
                out_img = Image.fromarray((out.numpy().transpose(1,2,0) * 255).astype('uint8'))
                out_path = os.path.join(out_dir, os.path.basename(img_path))
                out_img.save(out_path)
                logger.info("UPSCALE: processed %s -> %s", img_path, out_path)
            except Exception as e:
                logger.error("UPSCALE: failed to process %s: %s", img_path, e)
        logger.info("UPSCALE completed in %.3fs", time.perf_counter() - start)
        return out_dir
    except Exception as exc:
        logger.error("UPSCALE: error during upscaling: %s", exc)
        return input_dir


@tool
def assemble_pdf(*, image_dir: str, output_pdf: str = "output.pdf") -> Optional[str]:
    """Assemble all images in ``image_dir`` into a PDF and place it under ``output/<song>/<artist>/<key>/``."""
    start = time.perf_counter()
    try:
        images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        if not images:
            logger.error("ASSEMBLER: no images found in %s", image_dir)
            return None
        # Compose final PDF path using metadata
        meta = SCRAPE_METADATA.copy()
        title = meta.get('title', '')
        artist = meta.get('artist', '')
        key_meta = meta.get('key', '')
        instrument_meta = meta.get('instrument', '')
        def _sanitize(val: str) -> str:
            import re
            return re.sub(r"[^A-Za-z0-9]+", "_", val.strip()).strip("_")
        if title and artist and key_meta and instrument_meta:
            title_dir = _sanitize(title)
            artist_dir = _sanitize(artist)
            key_dir = _sanitize(key_meta)
            instr_dir = _sanitize(instrument_meta)
            pdf_root = os.path.join(os.getcwd(), "output")
            pdf_dir = os.path.join(pdf_root, title_dir, artist_dir, key_dir)
            os.makedirs(pdf_dir, exist_ok=True)
            file_title = _sanitize(title.lower())
            pdf_name = f"{file_title}_{instr_dir}_{key_dir}.pdf"
            final_pdf = os.path.join(pdf_dir, pdf_name)
        else:
            final_pdf = output_pdf
            os.makedirs(os.path.dirname(final_pdf) or '.', exist_ok=True)
        c = canvas.Canvas(final_pdf, pagesize=letter)
        width, height = letter
        for fname in images:
            c.drawImage(os.path.join(image_dir, fname), 0, 0, width=width, height=height, preserveAspectRatio=True)
            c.showPage()
        c.save()
        logger.info("ASSEMBLER: wrote %d pages to %s", len(images), final_pdf)
        logger.info("ASSEMBLER completed in %.3fs", time.perf_counter() - start)
        return final_pdf
    except Exception as exc:
        logger.error("ASSEMBLER: error assembling PDF: %s", exc)
        return None
    finally:
        _cleanup_temp_dirs()