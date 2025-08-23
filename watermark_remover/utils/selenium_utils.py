"""Selenium helpers and XPath definitions for Watermark Remover Agent.

This module centralises all XPaths used to navigate the PraiseCharts website and
provides utility functions that wrap common Selenium operations such as clicking
elements, locating elements, and sending keys.  The helpers include full page
screenshots with bounding boxes and structured logging of each click attempt.

The `xpaths` dictionary defines string templates for locating elements.  See the
original Watermark Remover project for full context.
"""

from __future__ import annotations

import threading
import time
import os
from typing import Any, Optional

from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore
from selenium.common.exceptions import (  # type: ignore
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from PIL import Image, ImageDraw  # type: ignore

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
# update the XPaths, be sure to update this mapping accordingly.
xpath_labels: dict[str, str] = {
    xpaths['chords_button']: "Chords & Lyrics",
    xpaths['orchestration_header']: "Orchestration",
    xpaths['key_button']: "Key",
    xpaths['parts_button']: "Instrument",
}


def _get_label(driver: Any, element: Any, xpath: str) -> str:
    """Try to determine a human readable label for an element."""
    try:
        label = (element.text or "").strip()
        if not label:
            label = (element.get_attribute("innerText") or "").strip()
        if not label:
            label = (element.get_attribute("textContent") or "").strip()
        if not label:
            label = (element.get_attribute("aria-label") or "").strip()
        if not label and driver is not None:
            try:
                label = (driver.execute_script("return arguments[0].textContent", element) or "").strip()
            except Exception:
                pass
    except Exception:
        label = ""
    if not label:
        label = xpath_labels.get(xpath, "")
    return label


class SeleniumHelper:
    """Utility methods for common Selenium operations with logging and screenshots."""

    @staticmethod
    def _screenshot_and_annotate(driver: Any, element: Any, screenshots_dir: str, label: str) -> str:
        """Capture a full page screenshot and draw a red bounding box around the element."""
        ts = int(time.time() * 1000)
        safe_label = ''.join(c if c.isalnum() else '_' for c in (label or 'element'))[:30] or 'element'
        filename = f"{ts}_{safe_label}.png"
        file_path = os.path.join(screenshots_dir, filename)
        # Save full page screenshot
        try:
            driver.save_screenshot(file_path)
            # Draw bounding box
            try:
                rect = element.rect
                # Use device pixel ratio to adjust coordinates
                dpr = 1
                try:
                    dpr = driver.execute_script("return window.devicePixelRatio || 1;") or 1
                except Exception:
                    dpr = 1
                scroll_x = 0
                scroll_y = 0
                try:
                    scroll_x = driver.execute_script("return window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0;") or 0
                    scroll_y = driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;") or 0
                except Exception:
                    scroll_x = 0
                    scroll_y = 0
                x = int((rect.get('x', 0) + scroll_x) * dpr)
                y = int((rect.get('y', 0) + scroll_y) * dpr)
                w = int(rect.get('width', 0) * dpr)
                h = int(rect.get('height', 0) * dpr)
                img = Image.open(file_path)
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                img.save(file_path)
            except Exception:
                pass
        except Exception:
            # If screenshot fails, return empty string
            return ""
        return os.path.relpath(file_path, os.getcwd())

    @staticmethod
    def click_element(driver: Any, xpath: str, timeout: float = 2, log_func: Optional[Any] = None) -> bool:
        """Attempt to click an element specified by an XPath.

        Returns True on success and False on failure.  Optionally logs debug
        information using `log_func`.  When logging, the helper will
        include the visible text of the element (if any) alongside the
        XPath so callers can see which button or link was pressed.
        """
        try:
            with selenium_lock:
                # Determine a human-readable initial label
                init_label = xpath_labels.get(xpath, "element")
                # Determine log and screenshot base directory
                base_dir = os.environ.get("WMRA_LOG_DIR", os.path.join(os.getcwd(), "logs"))
                screenshots_dir = os.path.join(base_dir, "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                # Capture pre-click screenshot for context
                pre_ts = int(time.time() * 1000)
                safe_init = ''.join(c if c.isalnum() else '_' for c in init_label)[:30] or 'element'
                pre_filename = f"{pre_ts}_{safe_init}.png"
                pre_path = os.path.join(screenshots_dir, pre_filename)
                try:
                    driver.save_screenshot(pre_path)
                except Exception:
                    pass
                screenshot_path = pre_path
                # Wait for element to be clickable
                if log_func:
                    log_func(f"[DEBUG] Waiting for element to be clickable: {xpath}")
                element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
                # Scroll element into view
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                    if log_func:
                        log_func(f"[DEBUG] Scrolled element into view: {xpath}")
                except Exception:
                    if log_func:
                        log_func("[DEBUG] Failed to scroll element into view")
                # Determine label using multiple strategies
                label = _get_label(driver, element, xpath)
                # Capture annotated screenshot
                annotated_path = SeleniumHelper._screenshot_and_annotate(driver, element, screenshots_dir, label)
                if annotated_path:
                    screenshot_path = annotated_path
                # Capture current URL
                url = ""
                try:
                    url = driver.current_url
                except Exception:
                    url = ""
                # Log the attempt
                if log_func:
                    msg_attempt = f"[DEBUG] Attempting click on element: '{label}' ({xpath})" if label else f"[DEBUG] Attempting click on element: {xpath}"
                    if url:
                        msg_attempt += f" at {url}"
                    try:
                        log_func(msg_attempt, extra={"button_text": label or "", "xpath": xpath, "url": url, "screenshot": screenshot_path})
                    except TypeError:
                        log_func(msg_attempt)
                # Perform the click
                element.click()
                # Log success with updated URL
                url_after = ""
                try:
                    url_after = driver.current_url
                except Exception:
                    url_after = ""
                if log_func:
                    msg_success = f"[DEBUG] Click successful on element: '{label}' ({xpath})" if label else f"[DEBUG] Click successful on element: {xpath}"
                    if url_after:
                        msg_success += f" at {url_after}"
                    try:
                        log_func(msg_success, extra={"button_text": label or "", "xpath": xpath, "url": url_after, "screenshot": screenshot_path})
                    except TypeError:
                        log_func(msg_success)
                return True
        except (ElementClickInterceptedException, NoSuchElementException, StaleElementReferenceException, TimeoutException):
            return False
        except Exception:
            return False

    @staticmethod
    def find_element(driver: Any, xpath: str, timeout: float = 2, log_func: Optional[Any] = None) -> Optional[Any]:
        """Find a single element by XPath.  Returns the element or None if not found."""
        try:
            with selenium_lock:
                if log_func:
                    log_func(f"[DEBUG] Searching for element: {xpath}")
                element = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
                if log_func:
                    log_func(f"[DEBUG] Element found: {xpath}")
                return element
        except (NoSuchElementException, TimeoutException):
            if log_func:
                log_func(f"[DEBUG] Element not found: {xpath}")
            return None

    @staticmethod
    def find_elements(driver: Any, xpath: str, timeout: float = 2, log_func: Optional[Any] = None) -> list[Any]:
        """Find multiple elements by XPath.  Returns an empty list if none are found."""
        try:
            with selenium_lock:
                if log_func:
                    log_func(f"[DEBUG] Searching for elements: {xpath}")
                elements = WebDriverWait(driver, timeout).until(EC.presence_of_all_elements_located((By.XPATH, xpath)))
                if log_func:
                    log_func(f"[DEBUG] Found {len(elements)} elements: {xpath}")
                return elements
        except (NoSuchElementException, TimeoutException):
            if log_func:
                log_func(f"[DEBUG] Elements not found: {xpath}")
            return []

    @staticmethod
    def send_keys(driver: Any, xpath: str, keys: str, timeout: float = 2, log_func: Optional[Any] = None) -> bool:
        """Send keys to an element specified by XPath."""
        try:
            with selenium_lock:
                if log_func:
                    log_func(f"[DEBUG] Waiting for element to send keys: {xpath}")
                element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
                # Scroll into view
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                except Exception:
                    pass
                element.clear()
                element.send_keys(keys)
                if log_func:
                    log_func(f"[DEBUG] Sent keys to element: {xpath}")
                return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Backwards compatibility
    # Many call sites in older versions of the agent invoked a method named
    # ``send_keys_to_element``.  Provide an alias that simply forwards to
    # :meth:`send_keys` for compatibility with existing tooling.
    @staticmethod
    def send_keys_to_element(driver: Any, xpath: str, keys: str, timeout: float = 2, log_func: Optional[Any] = None) -> bool:
        """Alias for :meth:`send_keys` to maintain compatibility with older code."""
        return SeleniumHelper.send_keys(driver, xpath, keys, timeout, log_func)