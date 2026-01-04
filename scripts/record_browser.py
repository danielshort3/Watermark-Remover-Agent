"""Record manual browser interactions for XPath discovery."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


LOGGER = logging.getLogger("wmra.recorder")


JS_INSTALL_RECORDER = r"""
(function() {
  if (window.__wmraRecorderInstalled) { return; }
  window.__wmraRecorderInstalled = true;
  window.__wmraEvents = window.__wmraEvents || [];
  function cssPath(el) {
    if (!el || !el.nodeType || el.nodeType !== Node.ELEMENT_NODE) { return ""; }
    var parts = [];
    while (el && el.nodeType === Node.ELEMENT_NODE) {
      var name = el.nodeName.toLowerCase();
      if (el.id) {
        parts.unshift(name + "#" + el.id);
        break;
      }
      var sib = el;
      var nth = 1;
      while (sib = sib.previousElementSibling) {
        if (sib.nodeName.toLowerCase() === name) { nth++; }
      }
      parts.unshift(name + ":nth-of-type(" + nth + ")");
      el = el.parentElement;
    }
    return parts.join(" > ");
  }
  function xpath(el) {
    if (!el || !el.nodeType || el.nodeType !== Node.ELEMENT_NODE) { return ""; }
    if (el.id) { return "//*[@id='" + el.id + "']"; }
    var parts = [];
    while (el && el.nodeType === Node.ELEMENT_NODE) {
      var index = 1;
      var sib = el.previousSibling;
      while (sib) {
        if (sib.nodeType === Node.ELEMENT_NODE && sib.nodeName === el.nodeName) {
          index++;
        }
        sib = sib.previousSibling;
      }
      parts.unshift(el.nodeName.toLowerCase() + "[" + index + "]");
      el = el.parentElement;
    }
    return "/" + parts.join("/");
  }
  function record(event, action) {
    try {
      var el = event.target;
      var text = "";
      try {
        text = (el.innerText || el.textContent || "").trim().slice(0, 160);
      } catch (e) { text = ""; }
      var value = "";
      try {
        value = (el.value !== undefined ? String(el.value) : "").slice(0, 160);
      } catch (e) { value = ""; }
      var outer = "";
      try {
        outer = (el.outerHTML || "").replace(/\s+/g, " ").slice(0, 300);
      } catch (e) { outer = ""; }
      window.__wmraEvents.push({
        ts: Date.now(),
        action: action,
        key: event.key || "",
        tag: (el && el.tagName) ? el.tagName.toLowerCase() : "",
        text: text,
        value: value,
        url: window.location.href,
        title: document.title || "",
        css: cssPath(el),
        xpath: xpath(el),
        outer_html: outer
      });
    } catch (e) {}
  }
  document.addEventListener("click", function(e) { record(e, "click"); }, true);
  document.addEventListener("input", function(e) { record(e, "input"); }, true);
  document.addEventListener("change", function(e) { record(e, "change"); }, true);
  document.addEventListener("keydown", function(e) {
    if (e.key === "Enter") { record(e, "keydown"); }
  }, true);
})();
"""


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_bool(value: bool | None) -> bool:
    return bool(value)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _start_driver(headless: bool, maximize: bool) -> Any:
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.service import Service  # type: ignore
        from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(f"Selenium dependencies not available: {exc}") from exc

    options = webdriver.ChromeOptions()
    if headless:
        try:
            options.add_argument("--headless=new")
        except Exception:
            options.add_argument("--headless")
    elif maximize:
        options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")

    for env_name in ("SELENIUM_BINARY", "CHROME_BINARY", "GOOGLE_CHROME_SHIM"):
        bin_override = os.environ.get(env_name)
        if bin_override and os.path.isfile(bin_override):
            options.binary_location = bin_override
            break

    for candidate in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if os.path.isfile(candidate):
            options.binary_location = candidate
            break

    try:
        return webdriver.Chrome(options=options)
    except Exception:
        pass

    driver_path_candidates = [
        "/usr/bin/chromedriver",
        "/usr/lib/chromium-browser/chromedriver",
        "/usr/lib/chromium/chromedriver",
    ]
    for candidate_path in driver_path_candidates:
        if os.path.isfile(candidate_path):
            try:
                service = Service(candidate_path)
                return webdriver.Chrome(service=service, options=options)
            except Exception:
                continue

    try:
        drv_path = ChromeDriverManager().install()
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve chromedriver: {exc}") from exc
    return webdriver.Chrome(service=Service(drv_path), options=options)


def _install_recorder(driver: Any) -> None:
    try:
        driver.execute_script(JS_INSTALL_RECORDER)
    except Exception as exc:
        LOGGER.debug("Recorder injection failed: %s", exc)


def _drain_events(driver: Any) -> List[Dict[str, Any]]:
    script = "var ev = window.__wmraEvents || []; window.__wmraEvents = []; return ev;"
    try:
        return driver.execute_script(script) or []
    except Exception:
        return []


def _write_event_summary(path: Path, event: Dict[str, Any]) -> None:
    action = event.get("action") or ""
    xpath = event.get("xpath") or ""
    css = event.get("css") or ""
    text = event.get("text") or ""
    value = event.get("value") or ""
    url = event.get("url") or ""
    line = (
        f"{event.get('event_id', ''):>4} {action:<7} xpath={xpath} css={css} "
        f"text={text!r} value={value!r} url={url}"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record browser actions for XPath discovery.")
    parser.add_argument("--url", default="https://www.praisecharts.com/search")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--no-screenshot", action="store_true")
    parser.add_argument("--start-maximized", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    out_dir = Path(os.getcwd()) / "output" / "recordings" / time.strftime("%Y%m%d_%H%M%S")
    screenshots_dir = out_dir / "screenshots"
    _ensure_dir(out_dir)
    if not args.no_screenshot:
        _ensure_dir(screenshots_dir)

    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "events.txt"
    summary_path.write_text("", encoding="utf-8")

    driver = _start_driver(headless=_safe_bool(args.headless), maximize=_safe_bool(args.start_maximized))
    LOGGER.info("Recording to %s", out_dir)
    LOGGER.info("Opening %s", args.url)
    driver.get(args.url)
    _install_recorder(driver)

    stop_event = threading.Event()

    def _wait_for_stop() -> None:
        try:
            input("Press Enter to stop recording...\n")
        except Exception:
            pass
        stop_event.set()

    threading.Thread(target=_wait_for_stop, daemon=True).start()

    event_id = 0
    start_ts = time.time()
    try:
        while True:
            if args.duration and (time.time() - start_ts) >= args.duration:
                break
            if stop_event.is_set():
                break
            _install_recorder(driver)
            events = _drain_events(driver)
            for ev in events:
                event_id += 1
                ev["event_id"] = event_id
                ev["ts_local"] = time.time()
                screenshot_path = ""
                if not args.no_screenshot:
                    screenshot_path = str(screenshots_dir / f"{event_id:04d}_{ev.get('action','event')}.png")
                    try:
                        driver.save_screenshot(screenshot_path)
                    except Exception:
                        screenshot_path = ""
                if screenshot_path:
                    ev["screenshot"] = screenshot_path
                with events_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev) + "\n")
                _write_event_summary(summary_path, ev)
            time.sleep(max(args.poll, 0.1))
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    LOGGER.info("Recording stopped. Events: %d", event_id)
    LOGGER.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
