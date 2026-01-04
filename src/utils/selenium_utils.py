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
# Moved to src/utils for shared helpers.

import random
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
    'search_bar': '//*[@id="search-input-wrap"]//input',
    'songs_parent': '//*[@id="page-wrapper"]//app-page-search//app-search//div',
    'song_title': './/h5',
    'song_text3': './/span/span',
    'song_text2': './/span',
    'song_image': './/app-product-audio-preview-image//img',
    'click_song': '//*[@id="page-wrapper"]//app-page-search//app-product-list-item[{index}]//a//div',
    'chords_button': '//*[@id="page-wrapper"]//app-product-sheet-selector//button[.//div[contains(normalize-space(.), "Chords")]]',
    # Modal product list shown after selecting "Chords & Lyrics".
    'product_modal_items': '//*[@id="products-sheet-music"]//h3',
    # Avoid clicking orchestration sections labelled with "Finale".  We use
    # translate() to perform a case‑insensitive check for 'orchestration'
    # and exclude ancestors whose text contains 'finale'.
    'orchestration_header': "//app-product-page//h3[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'orchestration')]/ancestor::div[4][not(contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'finale'))]",
    'key_button': '//*[@id="page-wrapper"]//app-product-selector-key//button[contains(@class, "dropdown-toggle")]',
    'key_parent': '//*[@id="page-wrapper"]//app-product-selector-key//ul',
    'parts_button': '//*[@id="page-wrapper"]//app-product-sheet-selector//button[contains(@class, "type-selector")]',
    'parts_parent': '//*[@id="page-wrapper"]//app-product-sheet-selector//button[contains(@class, "type-selector")]/following::ul[1]',
    'parts_list': '//*[@id="page-wrapper"]//app-product-sheet-selector//button[contains(@class, "type-selector")]/following::ul[1]/li/button',
    'image_element': '//*[@id="preview-sheets"]//img',
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


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _capture_settings() -> tuple[bool, bool, bool]:
    save_screens = _env_truthy(os.environ.get("WMRA_SAVE_SCREENSHOTS", "0"))
    save_html = _env_truthy(os.environ.get("WMRA_SAVE_HTML", "0"))
    errors_only = _env_truthy(os.environ.get("WMRA_SAVE_ON_ERROR_ONLY", "1"))
    return save_screens, save_html, errors_only


def _capture_artifacts(
    driver: Any,
    *,
    label: str,
    xpath: str,
    element: Any | None,
    save_screens: bool,
    save_html: bool,
    suffix: str = "",
) -> tuple[str, str]:
    screenshot_path = ""
    html_path = ""
    if not save_screens and not save_html:
        return screenshot_path, html_path
    try:
        base_dir = (
            os.environ.get("WMRA_PREVIEW_DIR")
            or os.environ.get("WMRA_LOG_DIR")
            or os.path.join(os.getcwd(), "output")
        )
        ts = int(time.time() * 1000)
        safe_label = "".join(c if c.isalnum() else "_" for c in (label or xpath or "element"))[:30]
        tag = f"{ts}_{safe_label}"
        if suffix:
            tag = f"{tag}_{suffix}"
    except Exception:
        base_dir = os.path.join(os.getcwd(), "output")
        tag = str(int(time.time() * 1000))

    if save_screens:
        try:
            out_dir = os.path.join(base_dir, "screenshots")
            os.makedirs(out_dir, exist_ok=True)
            file_path = os.path.join(out_dir, f"{tag}.png")
            driver.save_screenshot(file_path)
            if element is not None:
                try:
                    rect = element.rect
                    x = rect.get("x", 0) or 0
                    y = rect.get("y", 0) or 0
                    w = rect.get("width", 0) or 0
                    h = rect.get("height", 0) or 0
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
                    dpr = 1
                    try:
                        dpr = driver.execute_script("return window.devicePixelRatio || 1;") or 1
                    except Exception:
                        dpr = 1
                    x_pix = int((x + scroll_x) * dpr)
                    y_pix = int((y + scroll_y) * dpr)
                    w_pix = int(w * dpr)
                    h_pix = int(h * dpr)
                    img = Image.open(file_path)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(
                        [x_pix, y_pix, x_pix + w_pix, y_pix + h_pix],
                        outline=(255, 0, 0),
                        width=3,
                    )
                    img.save(file_path)
                except Exception:
                    pass
            screenshot_path = os.path.relpath(file_path, os.getcwd())
        except Exception:
            screenshot_path = ""

    if save_html:
        try:
            out_dir = os.path.join(base_dir, "html")
            os.makedirs(out_dir, exist_ok=True)
            file_path = os.path.join(out_dir, f"{tag}.html")
            html = driver.page_source or ""
            with open(file_path, "w", encoding="utf-8", errors="ignore") as handle:
                handle.write(html)
            html_path = os.path.relpath(file_path, os.getcwd())
        except Exception:
            html_path = ""

    return screenshot_path, html_path


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
        element = None
        label = ""
        screenshot_path = ""
        html_path = ""
        save_screens, save_html, errors_only = _capture_settings()
        capture_enabled = save_screens or save_html
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

                # ------------------------------------------------------------------
                # Special handling for the orchestration header: avoid clicking
                # if the containing block includes the word "Finale".  PraiseCharts
                # uses labels like "Finale: Orchestration" to denote Finale-only
                # orchestrations, which do not contain the desired sheet music.  If
                # such a label is detected, skip the click entirely.  Use
                # translate() via Selenium attributes to perform a case-insensitive
                # search.
                if xpath == xpaths.get('orchestration_header'):
                    try:
                        # Attempt to extract inner text directly from the element.  If
                        # this fails, use the label we already computed.
                        block_text = (element.text or "").strip()
                        if not block_text:
                            block_text = (element.get_attribute("innerText") or "").strip()
                    except Exception:
                        block_text = label or ""
                    # Check for 'finale' case-insensitively
                    if block_text and 'finale' in block_text.lower():
                        if log_func:
                            log_func("[DEBUG] Skipping orchestration click because block contains 'Finale'")
                        return False
                # Capture the current URL for context.  This helps when
                # reviewing logs to understand which page the click is
                # occurring on.  If obtaining the URL fails, leave it
                # blank.
                url = ""
                try:
                    url = driver.current_url
                except Exception:
                    url = ""

                if capture_enabled and not errors_only:
                    screenshot_path, html_path = _capture_artifacts(
                        driver,
                        label=label,
                        xpath=xpath,
                        element=element,
                        save_screens=save_screens,
                        save_html=save_html,
                    )
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
                        log_func(
                            msg_attempt,
                            extra={
                                "button_text": label or "",
                                "xpath": xpath,
                                "url": url,
                                "screenshot": screenshot_path,
                                "html": html_path,
                            },
                        )
                    except TypeError:
                        log_func(msg_attempt)
                # Add a small jitter so actions are not uniformly spaced.
                try:
                    time.sleep(random.uniform(0.0, 1.0))
                except Exception:
                    pass
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
                                "html": html_path,
                            },
                        )
                    except TypeError:
                        log_func(msg_success)
            return True
        except ElementClickInterceptedException as e:
            # If another element is in front, attempt fallback via JavaScript
            screenshot_path = ""
            html_path = ""
            if capture_enabled:
                screenshot_path, html_path = _capture_artifacts(
                    driver,
                    label=label or "",
                    xpath=xpath,
                    element=element,
                    save_screens=save_screens,
                    save_html=save_html,
                    suffix="error",
                )
            if log_func:
                msg = f"[DEBUG] Element click intercepted at xpath: {xpath} - {str(e)}"
                try:
                    log_func(
                        msg,
                        extra={
                            "button_text": label or "",
                            "xpath": xpath,
                            "url": "",
                            "screenshot": screenshot_path,
                            "html": html_path,
                        },
                    )
                except TypeError:
                    log_func(msg)
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
            screenshot_path = ""
            html_path = ""
            if capture_enabled:
                screenshot_path, html_path = _capture_artifacts(
                    driver,
                    label=label or "",
                    xpath=xpath,
                    element=element,
                    save_screens=save_screens,
                    save_html=save_html,
                    suffix="error",
                )
            if log_func:
                msg = f"Error clicking element at xpath: {xpath} - {str(e)}"
                try:
                    log_func(
                        msg,
                        extra={
                            "button_text": label or "",
                            "xpath": xpath,
                            "url": "",
                            "screenshot": screenshot_path,
                            "html": html_path,
                        },
                    )
                except TypeError:
                    log_func(msg)
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
        element = None
        screenshot_path = ""
        html_path = ""
        save_screens, save_html, errors_only = _capture_settings()
        capture_enabled = save_screens or save_html
        try:
            with selenium_lock:
                element = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                if capture_enabled and not errors_only:
                    screenshot_path, html_path = _capture_artifacts(
                        driver,
                        label="send_keys",
                        xpath=xpath,
                        element=element,
                        save_screens=save_screens,
                        save_html=save_html,
                    )
                # Add a small jitter so actions are not uniformly spaced.
                try:
                    time.sleep(random.uniform(0.0, 1.0))
                except Exception:
                    pass
                element.clear()
                element.send_keys(keys)
            return True
        except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
            screenshot_path = ""
            html_path = ""
            if capture_enabled:
                screenshot_path, html_path = _capture_artifacts(
                    driver,
                    label="send_keys",
                    xpath=xpath,
                    element=element,
                    save_screens=save_screens,
                    save_html=save_html,
                    suffix="error",
                )
            if log_func:
                msg = f"Error sending keys to element at xpath: {xpath} - {str(e)}"
                try:
                    log_func(
                        msg,
                        extra={
                            "button_text": "",
                            "xpath": xpath,
                            "url": "",
                            "screenshot": screenshot_path,
                            "html": html_path,
                        },
                    )
                except TypeError:
                    log_func(msg)
            return False

    @staticmethod
    def click_dynamic_element(driver: Any, xpath_template: str, index: int, timeout: float = 2, log_func: Any | None = None) -> bool:
        """Click an element whose XPath includes a positional index."""
        xpath = xpath_template.format(index=index)
        return SeleniumHelper.click_element(driver, xpath, timeout=timeout, log_func=log_func)
