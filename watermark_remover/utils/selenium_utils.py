"""Selenium helpers and XPath definitions from the original Watermark Remover.

This module is copied from the `watermark_remover/download/selenium_utils.py` file in
the Watermark Remover project.  It centralises all XPaths used to navigate the
PraiseCharts website and provides utility functions that wrap common Selenium
operations (clicking, locating elements, sending keys) with appropriate
waits, scrolling and error handling.  Importing this module allows the agent
version of the Watermark Remover to reuse the exact same navigation logic and
selectors as the original GUI application.

The `xpaths` dictionary defines string templates for locating elements.  See
the original project for full context.  The `SeleniumHelper` class exposes
static methods that operate on a Selenium WebDriver instance and perform
actions such as clicking elements, finding elements, and sending keys.  Each
method accepts an optional `log_func` callback that can be used to capture
debug messages.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from selenium.webdriver.common.by import By  # type: ignore
from PIL import Image, ImageDraw  # type: ignore
import os
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore


# Lock to serialize Selenium operations across threads
selenium_lock = threading.Lock()

# Centralised dictionary of XPaths used by the application
xpaths: dict[str, str] = {
    'search_bar': '//*[@id="search-input-wrap"]/input',
    'songs_parent': '//*[@id="page-wrapper"]/ion-router-outlet/app-page-search/ion-content/div/div/div/app-search/div',
    'song_title': './div/a/div/h5',
    'song_text3': './div/a/div/span/span',
    'song_text2': './div/a/div/span',
    'song_image': './div/div[1]/div/app-product-audio-preview-image/div/img',
    'click_song': '//*[@id="page-wrapper"]/ion-router-outlet/app-page-search/ion-content/div/div/div/app-search/div/app-product-list-item[{index}]/div/a/div',
    'chords_button': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[1]/button',
    'orchestration_header': "//h3[contains(text(), 'Orchestration')]/ancestor::div[4]",
    'key_button': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[3]/app-product-selector-key/div/button',
    'key_parent': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[3]/app-product-selector-key/div/ul',
    'parts_button': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[2]/div/button',
    'parts_parent': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content/div/div/div[3]/div/div[1]/div[2]/div[1]/app-product-sheet-selector/div/div[2]/div/ul',
    'parts_list': '//*[@id="page-wrapper"]/ion-router-outlet/app-product-page/ion-content//ul/li/button',
    'image_element': '//*[@id="preview-sheets"]/div/div[1]/div/img',
    'next_button': "//button[contains(@class, 'sheet-nav-gradient-button-right')]",
}

# Mapping from specific XPaths to human‑readable labels.  These labels are
# used when the element itself has no visible text.  The keys in this
# dictionary should match the full XPath strings defined above.  If you
# update the XPaths, be sure to update this mapping accordingly.  These
# labels are guesses based on the PraiseCharts UI and may be adjusted
# to reflect the actual button names.
xpath_labels: dict[str, str] = {}

# Populate the mapping after xpaths is defined.  This uses the actual
# XPath strings, not the keys in the xpaths dict.  If any of these
# xpaths are changed, this mapping should be updated accordingly.
xpath_labels[xpaths['chords_button']] = "Chords & Lyrics"
xpath_labels[xpaths['orchestration_header']] = "Orchestration"
xpath_labels[xpaths['key_button']] = "Key"
xpath_labels[xpaths['parts_button']] = "Instrument"


class SeleniumHelper:
    """Utility methods for common Selenium operations.

    These helpers wrap Selenium WebDriver calls with appropriate waits and
    scrolling to improve reliability.  Each method acquires a global lock
    to avoid concurrent interactions with the browser, making them safe to
    call from multiple threads.
    """

    @staticmethod
    def click_element(driver: Any, xpath: str, timeout: float = 2, log_func: Any | None = None) -> bool:
        """Attempt to click an element specified by an XPath.

        Returns True on success and False on failure.  Optionally logs debug
        information using `log_func`.  When logging, the helper will
        include the visible text of the element (if any) alongside the
        XPath so callers can see which button or link was pressed.
        """
        try:
            with selenium_lock:
                # Wait for the element to be clickable
                if log_func:
                    log_func(f"[DEBUG] Waiting for element to be clickable: {xpath}")
                element = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                # Scroll element into view
                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", element
                    )
                    if log_func:
                        log_func(f"[DEBUG] Scrolled element into view: {xpath}")
                except Exception:
                    if log_func:
                        log_func("[DEBUG] Failed to scroll element into view")
                # Determine a human‑readable label for the element.  Some
                # elements may not have visible text directly attached to the
                # clickable node.  Try multiple strategies to extract
                # meaningful text.  Fall back to an empty string if all
                # attempts yield nothing.
                label = ""
                try:
                    # Strategy 1: element.text property
                    label = (element.text or "").strip()
                    # Strategy 2: get innerText attribute
                    if not label:
                        label = (element.get_attribute("innerText") or "").strip()
                    # Strategy 3: get textContent attribute (covers some
                    # elements that store text only in textContent)
                    if not label:
                        label = (element.get_attribute("textContent") or "").strip()
                    # Strategy 4: get aria-label for accessibility
                    if not label:
                        label = (element.get_attribute("aria-label") or "").strip()
                    # Strategy 5: execute JS to get node's text content
                    if not label and driver is not None:
                        try:
                            label = (
                                driver.execute_script(
                                    "return arguments[0].textContent", element
                                )
                                or ""
                            ).strip()
                        except Exception:
                            # ignore JS errors
                            pass
                except Exception:
                    label = ""

                # If no label was discovered, fall back to a predefined
                # mapping of XPath strings to labels.  This allows us to
                # provide human‑readable names for buttons that do not have
                # visible text (e.g. icons or dropdown toggles).  See
                # `xpath_labels` defined at module level.
                if not label:
                    mapped = xpath_labels.get(xpath)
                    if mapped:
                        label = mapped
                # Capture the current URL for context.  This helps when
                # reviewing logs to understand which page the click is
                # occurring on.  If obtaining the URL fails, leave it
                # blank.
                url = ""
                try:
                    url = driver.current_url
                except Exception:
                    url = ""

                # Prepare a placeholder for the screenshot path.  This
                # variable will be populated before the actual click and
                # reused when logging the success message.  Define it
                # here to ensure it exists across the nested scopes.
                screenshot_path = ""
                # Log the click attempt with the label, xpath and URL.  Pass the
                # structured data via the extra dict so that the CSV handler can
                # populate separate columns for button_text, xpath and url.
                if log_func:
                    # Build the human‑readable message
                    if label:
                        msg_attempt = f"[DEBUG] Attempting click on element: '{label}' ({xpath})"
                    else:
                        msg_attempt = f"[DEBUG] Attempting click on element: {xpath}"
                    if url:
                        msg_attempt += f" at {url}"
                    try:
                        # Prepare screenshot of the entire viewport to show where the click will occur
                        screenshot_path = ""
                        try:
                            # Determine the base directory for logs/screenshots.  Use the
                            # WMRA_LOG_DIR environment variable if set by tools.py; fall
                            # back to a local ``output`` directory otherwise.
                            # Use the WMRA_LOG_DIR from tools.py if available, otherwise
                            # fall back to ``output/logs``.  This ensures
                            # screenshots live alongside other run artefacts.
                            base_dir = os.environ.get(
                                "WMRA_LOG_DIR",
                                os.path.join(os.getcwd(), "output", "logs"),
                            )
                            out_dir = os.path.join(base_dir, "screenshots")
                            os.makedirs(out_dir, exist_ok=True)
                            # Build a filename using the label and timestamp
                            ts = int(time.time() * 1000)
                            safe_label = ''.join(c if c.isalnum() else '_' for c in (label or 'element'))[:30]
                            # Prepend the timestamp to the filename so that
                            # screenshots sort in chronological order.  The
                            # timestamp is placed at the beginning of the
                            # filename, followed by a sanitized label.  This
                            # ensures all screenshots are unique and easy to
                            # order when reviewing a large number of images.
                            filename = f"{ts}_{safe_label}.png"
                            file_path = os.path.join(out_dir, filename)
                            # Capture full viewport screenshot
                            driver.save_screenshot(file_path)
                            # Highlight the element on the screenshot by drawing a red rectangle
                            try:
                                # Determine element bounds relative to the current viewport.
                                rect = element.rect  # x, y relative to the viewport in CSS pixels
                                x = rect.get('x', 0) or 0
                                y = rect.get('y', 0) or 0
                                w = rect.get('width', 0) or 0
                                h = rect.get('height', 0) or 0
                                # Retrieve scroll offsets to translate viewport coordinates into
                                # absolute page coordinates.  Without this adjustment the
                                # rectangle may be drawn at the wrong vertical position on
                                # full‑page screenshots when the page is scrolled.
                                scroll_x = 0
                                scroll_y = 0
                                try:
                                    scroll_x = driver.execute_script(
                                        "return window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0;"
                                    ) or 0
                                    scroll_y = driver.execute_script(
                                        "return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;"
                                    ) or 0
                                except Exception:
                                    scroll_x = 0
                                    scroll_y = 0
                                # Retrieve device pixel ratio to convert CSS pixels to screenshot pixels.
                                # On some platforms the screenshot resolution may differ from CSS pixel
                                # dimensions (e.g. high‑DPI displays).  Multiply coordinates by the
                                # ratio to align the annotation correctly.
                                dpr = 1
                                try:
                                    dpr = driver.execute_script("return window.devicePixelRatio || 1;") or 1
                                except Exception:
                                    dpr = 1
                                # Compute absolute coordinates in screenshot pixel space
                                x_pix = int((x + scroll_x) * dpr)
                                y_pix = int((y + scroll_y) * dpr)
                                w_pix = int(w * dpr)
                                h_pix = int(h * dpr)
                                # Open image and draw rectangle
                                img = Image.open(file_path)
                                draw = ImageDraw.Draw(img)
                                # Use red outline with thickness 3
                                draw.rectangle([
                                    x_pix,
                                    y_pix,
                                    x_pix + w_pix,
                                    y_pix + h_pix,
                                ], outline=(255, 0, 0), width=3)
                                img.save(file_path)
                            except Exception:
                                # If annotation fails, continue with plain screenshot
                                pass
                            # Store relative path from current working directory
                            screenshot_path = os.path.relpath(file_path, os.getcwd())
                        except Exception:
                            screenshot_path = ""
                        # Log with structured data, including the screenshot path
                        log_func(
                            msg_attempt,
                            extra={
                                "button_text": label or "",
                                "xpath": xpath,
                                "url": url,
                                "screenshot": screenshot_path,
                            },
                        )
                    except TypeError:
                        # Some log functions may not accept extra; fallback to plain log
                        log_func(msg_attempt)
                # Perform the click
                element.click()
                # Capture current URL again after the click, which may have
                # changed if navigation occurred
                try:
                    url_after = driver.current_url
                except Exception:
                    url_after = ""
                # Log success with structured data.  Use the same label and
                # xpath; update the URL to reflect the post‑click state.
                if log_func:
                    if label:
                        msg_success = f"[DEBUG] Click successful on element: '{label}' ({xpath})"
                    else:
                        msg_success = f"[DEBUG] Click successful on element: {xpath}"
                    if url_after:
                        msg_success += f" at {url_after}"
                    try:
                        log_func(
                            msg_success,
                            extra={
                                "button_text": label or "",
                                "xpath": xpath,
                                "url": url_after,
                                "screenshot": screenshot_path,
                            },
                        )
                    except TypeError:
                        log_func(msg_success)
            return True
        except ElementClickInterceptedException as e:
            # If another element is in front, attempt fallback via JavaScript
            if log_func:
                log_func(
                    f"[DEBUG] Element click intercepted at xpath: {xpath} - {str(e)}"
                )
            try:
                with selenium_lock:
                    driver.execute_script("arguments[0].click();", element)
                if log_func:
                    if label:
                        log_func(
                            f"[DEBUG] JavaScript click successful on element: '{label}' ({xpath})"
                        )
                    else:
                        log_func(
                            f"[DEBUG] JavaScript click successful on element: {xpath}"
                        )
                return True
            except Exception as js_ex:
                if log_func:
                    log_func(
                        f"[DEBUG] JavaScript click failed on element: {xpath} - {str(js_ex)}"
                    )
                return False
        except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
            if log_func:
                log_func(f"Error clicking element at xpath: {xpath} - {str(e)}")
            return False

    @staticmethod
    def find_element(driver: Any, xpath: str, timeout: float = 2, log_func: Any | None = None):
        """Find a single element by XPath.  Returns the element or None if not found."""
        try:
            with selenium_lock:
                if log_func:
                    log_func(f"[DEBUG] Searching for element: {xpath}")
                element = WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                if log_func:
                    log_func(f"[DEBUG] Element found: {xpath}")
            return element
        except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
            if log_func:
                log_func(f"Error finding element at xpath: {xpath} - {str(e)}")
            return None

    @staticmethod
    def find_elements(driver: Any, xpath: str, timeout: float = 2, log_func: Any | None = None) -> list[Any]:
        """Find all elements matching the XPath.  Returns an empty list if none found."""
        try:
            with selenium_lock:
                if log_func:
                    log_func(f"[DEBUG] Searching for elements: {xpath}")
                elements = WebDriverWait(driver, timeout).until(
                    EC.presence_of_all_elements_located((By.XPATH, xpath))
                )
                if log_func:
                    log_func(
                        f"[DEBUG] Found {len(elements)} elements for xpath: {xpath}"
                    )
            return elements
        except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
            if log_func:
                log_func(f"Error finding elements at xpath: {xpath} - {str(e)}")
            return []

    @staticmethod
    def send_keys_to_element(driver: Any, xpath: str, keys: str, timeout: float = 2, log_func: Any | None = None) -> bool:
        """Send keys to an element specified by XPath.  Returns True on success."""
        try:
            with selenium_lock:
                element = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                element.clear()
                element.send_keys(keys)
            return True
        except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
            if log_func:
                log_func(f"Error sending keys to element at xpath: {xpath} - {str(e)}")
            return False

    @staticmethod
    def click_dynamic_element(driver: Any, xpath_template: str, index: int, timeout: float = 2, log_func: Any | None = None) -> bool:
        """Click an element whose XPath includes a positional index."""
        xpath = xpath_template.format(index=index)
        return SeleniumHelper.click_element(driver, xpath, timeout=timeout, log_func=log_func)