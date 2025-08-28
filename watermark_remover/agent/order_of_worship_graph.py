"""LangGraph pipeline for processing an order of worship PDF with extensive debugging
and LLM-only natural-language handling (no heuristic NLP).

Key changes:
- Adds robust debugging (run IDs, per-node timings, structured errors, persisted artifacts).
- Uses ONLY the LLM (run_instruction) to parse user instructions and extract song metadata.
- Removes all regex/rule-based NLP parsing for instructions and songs.

Debug artifacts are written to:
  output/debug/orders/<order_folder or 'unknown_date'>/<run_id>/
including:
  - run.log (full run-level logging)
  - parser_prompt.txt / parser_raw_output.txt / parser_output.json
  - pdf_text.txt
  - song_extractor_prompt.txt / song_extractor_raw_output.txt / songs_llm_output.json
  - songs_with_instruments.json
  - per-song step IO (e.g., song_01_scrape_input.json, song_01_scrape_output.txt, etc.)
  - final_state.json, errors.json, timings.json

Toggle console+file debugging:
  - Environment: ORDER_DEBUG=1 (default) or ORDER_DEBUG=0
  - Or pass in the initial state: {"debug": True} (overrides env)
"""

from __future__ import annotations

import re
import os
import json
import time
import uuid
import traceback
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path

from langgraph.graph import StateGraph, START, END

# Import tools and utilities from your project
from watermark_remover.agent.tools import sanitize_title, SCRAPE_METADATA, TEMP_DIRS
from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)

# Attempt to import the Ollama-backed LLM runner. If unavailable, we DO NOT fall back.
try:
    from watermark_remover.agent.graph_ollama import run_instruction  # type: ignore
except Exception:
    run_instruction = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging / Debugging Utilities
# ---------------------------------------------------------------------------

LOGGER_NAME = "order_of_worship"
_logger = logging.getLogger(LOGGER_NAME)

def _env_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() not in ("", "0", "false", "no", "off")

def _debug_enabled(state: Dict[str, Any]) -> bool:
    if isinstance(state.get("debug"), bool):
        return state["debug"]
    return _env_truthy(os.getenv("ORDER_DEBUG", "1"))

def _ensure_logger_configured(level: int) -> None:
    # Avoid duplicate handlers if module reloaded
    if not _logger.handlers:
        _logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        _logger.addHandler(ch)

def _get_run_id(state: Dict[str, Any]) -> str:
    rid = state.get("run_id")
    if not rid:
        rid = uuid.uuid4().hex[:10]
        state["run_id"] = rid
    return rid

def _get_order_folder(state: Dict[str, Any]) -> str:
    # Compute from PDF if possible; fallback to state-provided or 'unknown'
    of = (state.get("order_folder") or "").strip()
    if of and of.lower() != "unknown" and re.fullmatch(r"\d{2}_\d{2}_\d{4}", of):
        # Already set/normalized
        try:
            return sanitize_title(of)
        except Exception:
            return of

    # Try to compute from the actual PDF text
    pdf_name = state.get("pdf_name")
    pdf_path = None
    if pdf_name:
        # Resolve path similar to extractor
        pdf_path = pdf_name
        if not os.path.isabs(pdf_path):
            root_dir = state.get("pdf_root") or state.get("input_dir") or "input"
            candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
            pdf_path = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), pdf_path)
        if os.path.exists(pdf_path):
            try:
                computed = _determine_order_folder_from_pdf(state, pdf_path)
                if computed and computed.lower() != "unknown":
                    return sanitize_title(computed)
            except Exception:
                pass

    # Last resort
    of = of or "unknown"
    try:
        return sanitize_title(of)
    except Exception:
        return of

def _debug_dir_for(state: Dict[str, Any]) -> Path:
    run_id = _get_run_id(state)
    order_folder = _get_order_folder(state)
    base = Path(os.getcwd()) / "output" / "debug" / "orders" / order_folder / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base

def _ensure_file_handler(state: Dict[str, Any], base_dir: Path) -> None:
    # Attach exactly one file handler per run
    log_path = base_dir / "run.log"
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_log_path", None) == str(log_path)
        for h in _logger.handlers
    )
    if not already:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        fh.setFormatter(fmt)
        setattr(fh, "_log_path", str(log_path))
        _logger.addHandler(fh)

def _write_text(base: Path, name: str, content: str) -> str:
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", errors="ignore")
    return str(p)

def _write_json(base: Path, name: str, obj: Any) -> str:
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)

def _record_error(state: Dict[str, Any], label: str, exc: BaseException) -> None:
    errs: List[Dict[str, Any]] = state.get("errors", [])
    info = {
        "label": label,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    errs.append(info)
    state["errors"] = errs
    if _debug_enabled(state):
        base = _debug_dir_for(state)
        _write_json(base, "errors.json", errs)
    _logger.exception("Error [%s]: %s", label, exc)

def _start_timer() -> float:
    return time.perf_counter()

def _stop_timer(start: float) -> float:
    return time.perf_counter() - start

def _record_timing(state: Dict[str, Any], segment: str, seconds: float) -> None:
    timings: Dict[str, float] = state.get("timings", {})
    timings[segment] = timings.get(segment, 0.0) + seconds
    state["timings"] = timings
    if _debug_enabled(state):
        base = _debug_dir_for(state)
        _write_json(base, "timings.json", timings)

MAX_LLM_RETRIES = int(os.getenv("ORDER_MAX_LLM_RETRIES", "2"))

def _extract_json_from_llm_output(raw: Any) -> Any:
    s = str(raw or "").strip()
    # If the assistant includes a <think>…</think> wrapper, drop the wrapper.
    end = s.rfind("</think>")
    if end != -1:
        s = s[end + len("</think>") :].lstrip()

    if raw is None:
        raise ValueError("LLM returned None (no content).")
    if isinstance(raw, (dict, list)):
        return raw

    # Try to extract the outermost JSON object or array
    lcb, rcb = s.find("{"), s.rfind("}")
    if lcb != -1 and rcb != -1 and rcb > lcb:
        try:
            return json.loads(s[lcb : rcb + 1])
        except Exception as e:
            # fall through to try array
            pass

    lsb, rsb = s.find("["), s.rfind("]")
    if lsb != -1 and rsb != -1 and rsb > lsb:
        try:
            return json.loads(s[lsb : rsb + 1])
        except Exception:
            pass

    # Last attempt: try the whole string (might already be pure JSON)
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output as JSON. First 200 chars: {s[:200]!r}") from e

def _run_llm_strict_json(
    state: Dict[str, Any],
    label: str,
    prompt: str,
    expected: str = "object",   # "object", "array", or "object_or_array"
) -> Any:
    """
    Calls run_instruction with retries and strict 'JSON only' guardrails.
    Persists each attempt's prompt and raw output to debug dir.
    Returns parsed JSON (dict or list). Raises ValueError after all retries fail.
    """
    if run_instruction is None:
        raise RuntimeError("LLM 'run_instruction' is unavailable.")

    attempts_log: List[Dict[str, Any]] = []
    base = _debug_dir_for(state) if _debug_enabled(state) else None

    # Build a stricter follow-up instruction used for retries
    strict_suffix = (
        "\n\nIMPORTANT:\n"
        "- Reply with STRICT JSON ONLY (no code fences, no prose).\n"
        f"- Expected top-level type: {expected}.\n"
        "- If uncertain or fields are missing, reply with an empty object {} "
        "or an empty array [] as appropriate.\n"
    )

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        attempt_label = f"{label}.attempt{attempt}"
        this_prompt = prompt if attempt == 1 else (prompt + strict_suffix)

        # Persist prompt
        if base is not None:
            _write_text(base, f"{label}_prompt.attempt{attempt}.txt", this_prompt)

        # Call LLM
        raw = None
        err_msg = None
        parsed: Any = None
        try:
            raw = run_instruction(this_prompt)
            # Persist raw
            if base is not None:
                _write_text(base, f"{label}_raw_output.attempt{attempt}.txt", str(raw))

            parsed = _extract_json_from_llm_output(raw)

            # Validate expected top-level type
            if expected == "object" and not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
            if expected == "array" and not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}.")
            if expected == "object_or_array" and not isinstance(parsed, (dict, list)):
                raise ValueError(f"Expected a JSON object or array, got {type(parsed).__name__}.")

            # Success
            attempts_log.append({"attempt": attempt, "status": "ok", "raw_len": len(str(raw))})
            break

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            attempts_log.append({"attempt": attempt, "status": "fail", "error": err_msg})
            if attempt == MAX_LLM_RETRIES:
                # Persist attempts summary
                if base is not None:
                    _write_json(base, f"{label}_attempts.json", attempts_log)
                raise

    # Persist attempts summary
    if base is not None:
        _write_json(base, f"{label}_attempts.json", attempts_log)

    return parsed


# ---------------------------------------------------------------------------
# PDF Text Extraction (not NLP; just I/O)
# ---------------------------------------------------------------------------

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore

def _read_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF using one of several fallback libraries."""
    if pdfplumber is not None:
        try:
            chunks: List[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    chunks.append(page.extract_text() or "")
            return "\n".join(chunks)
        except Exception:
            pass
    if PdfReader is not None:
        try:
            reader = PdfReader(pdf_path)  # type: ignore[call-arg]
            chunks: List[str] = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks)
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            txt = pdfminer_extract_text(pdf_path)  # type: ignore[call-arg]
            return txt or ""
        except Exception:
            pass
    raise RuntimeError(
        "Could not read PDF text. Please install pdfplumber, PyPDF2, or pdfminer.six."
    )



# ---------------------------------------------------------------------------
# Date of Service Extraction
# ---------------------------------------------------------------------------

_DATE_KEYWORDS = [
    "date of service",
    "service date",
    "worship date",
    "sunday",
    "saturday",
    "friday",
    "worship on",
    "worship for",
    "order of worship for",
    "date:"
]

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Common date regexes (US‑centric; we'll normalize to MM_DD_YYYY)
_DATE_PATTERNS = [
    # 01/05/2025 or 1/5/2025
    r"(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{4})",
    # 2025-01-05
    r"(?P<y>\d{4})[.-](?P<m>\d{1,2})[.-](?P<d>\d{1,2})",
    # January 5, 2025  or Jan 5, 2025
    r"(?P<mon>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(?P<y>\d{4})",
    # 5 January 2025 (UK style)
    r"(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))\s+(?P<y>\d{4})",
]

def _normalize_date_mm_dd_yyyy(y: int, m: int, d: int) -> str:
    return f"{m:02d}_{d:02d}_{y:04d}"

def _parse_month_name(mon: str) -> int | None:
    if not mon:
        return None
    return _MONTHS.get(mon.strip().lower())

def _try_parse_date_fragment(s: str) -> list[tuple[int,int,int,int,int]]:
    """Return list of (start, end, y, m, d) tuples for every date found in s."""
    found: list[tuple[int,int,int,int,int]] = []
    for pattern in _DATE_PATTERNS:
        for m in re.finditer(pattern, s, flags=re.IGNORECASE):
            gd = m.groupdict()
            try:
                if gd.get("mon"):
                    mnum = _parse_month_name(gd["mon"])
                    if not mnum:
                        continue
                    day = int(gd["d"])
                    year = int(gd["y"])
                    found.append((m.start(), m.end(), year, mnum, day))
                else:
                    year = int(gd["y"])
                    month = int(gd["m"])
                    day = int(gd["d"])
                    # Very naive sanity check
                    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
                        continue
                    found.append((m.start(), m.end(), year, month, day))
            except Exception:
                continue
    return found

def _score_date_candidate(text: str, pos_start: int, pos_end: int) -> int:
    """Heuristic score: closer to keywords gets higher score; date in title block gets slight boost."""
    window = 200  # characters to look around
    start = max(0, pos_start - window)
    end = min(len(text), pos_end + window)
    around = text[start:end].lower()
    score = 0
    for kw in _DATE_KEYWORDS:
        if kw in around:
            score += 5
    # Boost if line contains 'worship' or 'order of worship'
    line_start = text.rfind('\n', 0, pos_start) + 1
    line_end = text.find('\n', pos_end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end].lower()
    if "worship" in line:
        score += 2
    # Slight boost if date is within first 1k chars (cover page)
    if pos_start < 1000:
        score += 1
    return score

def _extract_service_date_heuristic(pdf_text: str) -> str | None:
    """Heuristically pick the most plausible service date and normalize to MM_DD_YYYY."""
    candidates = _try_parse_date_fragment(pdf_text)
    if not candidates:
        return None
    # Choose max score (ties broken by earliest in doc)
    best = None
    best_score = -10**9
    for start, end, y, m, d in candidates:
        score = _score_date_candidate(pdf_text, start, end)
        if score > best_score or (score == best_score and (best is None or start < best[0])):
            best = (start, end, y, m, d)
            best_score = score
    if best is None:
        return None
    _, _, y, m, d = best
    return _normalize_date_mm_dd_yyyy(y, m, d)

def _build_date_extractor_prompt(pdf_text: str) -> str:
    system = (
        "Extract the church 'Date of Service' from RAW text of an Order of Worship PDF. "
        "Return STRICT JSON with exactly one key: date. "
        "Format the value as MM_DD_YYYY (zero‑padded). "
        "If no date is present, return {\"date\": \"\"}. "
        "Prefer dates near phrases like 'Date of Service', 'Service Date', 'Order of Worship for', or weekday names."
    )
    return system + "\n\nRAW TEXT:\n" + pdf_text[:6000] + "\n\nReturn ONLY JSON."

def _extract_service_date_with_llm(state: Dict[str, Any], pdf_text: str) -> str | None:
    if run_instruction is None:
        return None
    try:
        prompt = _build_date_extractor_prompt(pdf_text)
        data = _run_llm_strict_json(state=state, label="date_extractor", prompt=prompt, expected="object")
        if isinstance(data, dict):
            val = (data.get("date") or "").strip()
            # Normalize if LLM returned a recognizable date in other format
            if re.fullmatch(r"\d{2}_\d{2}_\d{4}", val):
                return val
            # Try to re‑parse with our regex to make MM_DD_YYYY
            ff = _try_parse_date_fragment(val)
            if ff:
                _, _, y, m, d = ff[0]
                return _normalize_date_mm_dd_yyyy(y, m, d)
    except Exception as e:
        _record_error(state, "date_extractor.exception", e)
    return None

def _determine_order_folder_from_pdf(state: Dict[str, Any], pdf_path: str) -> str:
    """Read PDF text, then use LLM (preferred) or heuristics to produce MM_DD_YYYY, else 'unknown'."""
    try:
        raw = _read_pdf_text(pdf_path)
    except Exception as e:
        _record_error(state, "date_extractor.read_pdf_exception", e)
        raw = ""

    if _debug_enabled(state):
        base = _debug_dir_for(state)
        _write_text(base, "pdf_text.txt", raw)

    # First try LLM
    date = _extract_service_date_with_llm(state, raw)
    # Fallback to heuristics
    if not date:
        date = _extract_service_date_heuristic(raw)

    if not date:
        date = "unknown"

    # Persist decision for debugging
    if _debug_enabled(state):
        base = _debug_dir_for(state)
        _write_text(base, "date_of_service.txt", date)

    # Sanitize folder name
    try:
        return sanitize_title(date)
    except Exception:
        return date or "unknown"

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_PARSER_SYSTEM = (
    "You are a precise data extraction assistant. "
    "Given an instruction about processing an 'order of worship' PDF, you MUST return "
    "a single strict JSON object with exactly these keys: "
    "pdf_name (string, must include '.pdf'), "
    "default_instrument (string), "
    "overrides (object mapping zero-based integer indices to instrument strings), "
    "order_folder (string: folder name for this run). "
    "Do not include any explanation text. If something is missing, return an empty string or empty object for that key."
)

_SONG_EXTRACT_SYSTEM = (
    "You are a precise parser for 'order of worship' text. "
    "Input is RAW TEXT extracted from a PDF. Identify the list of songs and return a single strict JSON object "
    "mapping zero-based integer indices to objects with exactly: title (string), artist (string), key (string). "
    "If details cannot be found for a field, use an empty string. Do not fabricate songs not present in the text. "
    "Return only JSON, no extra commentary."
)

def _build_parser_prompt(instruction: str) -> str:
    return (
        _PARSER_SYSTEM
        + "\n\nInstruction:\n"
        + instruction
        + "\n\nReturn ONLY the JSON object."
    )

def _build_song_extractor_prompt(pdf_text: str) -> str:
    # We pass the raw text; LLM does all NLP extraction.
    return (
        _SONG_EXTRACT_SYSTEM
        + "\n\nRAW ORDER OF WORSHIP TEXT:\n"
        + pdf_text
        + "\n\nReturn ONLY the JSON object."
    )

# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-only parsing of the user instruction into pdf_name, default_instrument, overrides, order_folder."""
    # Configure logging based on debug flag
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _ensure_file_handler(new_state, base)
        _logger.debug("parser_node: starting with state keys: %s", list(new_state.keys()))

    instruction = (new_state.get("user_input") or "").strip()
    if not instruction:
        _logger.warning("parser_node: 'user_input' is empty.")
        _record_timing(new_state, "parser_node", _stop_timer(start))
        return new_state

    if run_instruction is None:
        err = RuntimeError("LLM 'run_instruction' is unavailable; cannot parse user_input.")
        _record_error(new_state, "parser_node.run_instruction_missing", err)
        _record_timing(new_state, "parser_node", _stop_timer(start))
        return new_state

    try:
        prompt = _build_parser_prompt(instruction)

        data = _run_llm_strict_json(
            state=new_state,
            label="parser",
            prompt=prompt,
            expected="object",
        )

        if not isinstance(data, dict):
            raise ValueError("Parser LLM did not return a JSON object.")

        pdf_name = (data.get("pdf_name") or "").strip()
        default_instrument = (data.get("default_instrument") or "").strip()
        overrides_raw = data.get("overrides") or {}
        order_folder_llm = (data.get("order_folder") or "").strip()

        # Compute order_folder (MM_DD_YYYY) from the PDF text (LLM preferred; fallback to heuristics)
        try:
            # Resolve a likely path to PDF
            pdf_path = pdf_name
            if not os.path.isabs(pdf_path):
                root_dir = new_state.get("pdf_root") or new_state.get("input_dir") or "input"
                candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
                pdf_path = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), pdf_path)
            order_folder_computed = _determine_order_folder_from_pdf(new_state, pdf_path) if os.path.exists(pdf_path) else "unknown"
        except Exception as e:
            _record_error(new_state, "parser_node.compute_order_folder_exception", e)
            order_folder_computed = "unknown"
        new_state["order_folder"] = order_folder_llm or order_folder_computed

        # Compute order_folder (MM_DD_YYYY) from the PDF text (LLM preferred; fallback to heuristics)
        try:
            # Resolve a likely path to PDF
            pdf_path = pdf_name
            if not os.path.isabs(pdf_path):
                root_dir = new_state.get("pdf_root") or new_state.get("input_dir") or "input"
                candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
                pdf_path = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), pdf_path)
            order_folder_computed = _determine_order_folder_from_pdf(new_state, pdf_path) if os.path.exists(pdf_path) else "unknown"
        except Exception as e:
            _record_error(new_state, "parser_node.compute_order_folder_exception", e)
            order_folder_computed = "unknown"
        new_state["order_folder"] = order_folder_llm or order_folder_computed

        # Convert overrides keys to ints if provided as strings
        overrides: Dict[int, str] = {}
        if isinstance(overrides_raw, dict):
            for k, v in overrides_raw.items():
                try:
                    idx = int(k)
                    overrides[idx] = (str(v) if v is not None else "").strip()
                except Exception:
                    # Keep it debuggable; skip invalid keys
                    pass

        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "parser_output.json", {
                "pdf_name": pdf_name,
                "default_instrument": default_instrument,
                "overrides": overrides,
                "order_folder": new_state.get("order_folder")
            })

        if pdf_name and not new_state.get("pdf_name"):
            new_state["pdf_name"] = pdf_name
        if default_instrument and not new_state.get("default_instrument"):
            new_state["default_instrument"] = default_instrument
        # Normalize any LLM/heuristic-provided order_folder already stored in state
        _of_val = new_state.get("order_folder")
        if _of_val:
            try:
                new_state["order_folder"] = sanitize_title(_of_val)
            except Exception:
                new_state["order_folder"] = _of_val

        merged_overrides: Dict[int, str] = dict(new_state.get("overrides", {}))
        merged_overrides.update(overrides)
        new_state["overrides"] = merged_overrides

    except Exception as e:
        _record_error(new_state, "parser_node.exception", e)

    finally:
        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "state_after_parser.json", {
                k: v for k, v in new_state.items() if k not in ("songs", "final_pdfs")
            })
        _record_timing(new_state, "parser_node", _stop_timer(start))

    return new_state


def extract_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-only extraction of songs from the PDF text; attaches instrument based on defaults and overrides."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _ensure_file_handler(new_state, base)
        _logger.debug("extract_songs_node: begin")

    pdf_name = new_state.get("pdf_name")
    default_instrument = (new_state.get("default_instrument") or "").strip()
    overrides: Dict[int, str] = new_state.get("overrides", {}) or {}

    if not pdf_name:
        _logger.warning("extract_songs_node: 'pdf_name' missing; no songs extracted.")
        new_state["songs"] = {}
        _record_timing(new_state, "extract_songs_node", _stop_timer(start))
        return new_state

    # Resolve a likely path; pure I/O (not NLP)
    pdf_path = pdf_name
    if not os.path.isabs(pdf_path):
        root_dir = new_state.get("pdf_root") or new_state.get("input_dir") or "input"
        candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
        pdf_path = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), pdf_path)

    # Ensure order_folder derived from PDF text (LLM preferred; fallback to heuristics)
    try:
        date_folder = _get_order_folder(new_state)
        new_state["order_folder"] = date_folder
    except Exception as e:
        _record_error(new_state, "extract_songs_node.ensure_order_folder_exception", e)

    if run_instruction is None:
        err = RuntimeError("LLM 'run_instruction' is unavailable; cannot extract songs from PDF text.")
        _record_error(new_state, "extract_songs_node.run_instruction_missing", err)
        new_state["songs"] = {}
        _record_timing(new_state, "extract_songs_node", _stop_timer(start))
        return new_state

    try:
        # Read raw PDF text and persist it for troubleshooting
        pdf_text = _read_pdf_text(pdf_path)
        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_text(base, "pdf_text.txt", pdf_text)

        # LLM prompt to extract songs (JSON only)
        prompt = _build_song_extractor_prompt(pdf_text)

        data = _run_llm_strict_json(
            state=new_state,
            label="song_extractor",
            prompt=prompt,
            expected="object_or_array",
        )

        # Accept dict or list; normalize to {idx: obj}
        if isinstance(data, list):
            normalized = {i: v for i, v in enumerate(data)}
        elif isinstance(data, dict):
            normalized = data
        else:
            raise ValueError(f"Unexpected JSON type from song extractor: {type(data).__name__}")

        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "songs_llm_output.json", normalized)

        # Normalize song entries + attach instrument (structure only)
        songs: Dict[int, Dict[str, Any]] = {}
        for k, v in normalized.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            title = (v.get("title") or "").strip()
            artist = (v.get("artist") or "").strip()
            key = (v.get("key") or "").strip()
            instrument = overrides.get(idx, default_instrument)
            songs[idx] = {
                "title": title,
                "artist": artist,
                "key": key,
                "instrument": instrument or "",
            }

        new_state["songs"] = songs

        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "songs_with_instruments.json", songs)

    except Exception as e:
        _record_error(new_state, "extract_songs_node.exception", e)
        new_state["songs"] = {}

    finally:
        _record_timing(new_state, "extract_songs_node", _stop_timer(start))

    return new_state


def process_songs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sequentially process songs via tools; capture step-by-step IO and errors for each song."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _ensure_file_handler(new_state, base)
        _logger.debug("process_songs_node: begin")

    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    final_pdfs: List[str | None] = []
    input_dir_override = new_state.get("input_dir", "data/samples")

    # We only use LLM-provided order_folder (if any); otherwise keep 'unknown_date'
    date_folder = _get_order_folder(new_state)

    for idx in sorted(songs.keys()):
        song = songs[idx]
        safe_title = sanitize_title(song.get("title", "Unknown Title"))
        per_song_prefix = f"song_{str(idx + 1).zfill(2)}"
        per_song_name = f"{per_song_prefix}_{safe_title}"

        try:
            # 1) SCRAPE
            scrape_input = {
                "title": song.get("title", ""),
                "instrument": song.get("instrument", ""),
                "key": song.get("key", ""),
                "input_dir": input_dir_override,
                "artist": song.get("artist", ""),
            }
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_scrape_input.json", scrape_input)

            try:
                download_dir = scrape_music.invoke(scrape_input)
            except Exception as e:
                download_dir = None
                _record_error(new_state, f"{per_song_name}_scrape_exception", e)

            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_scrape_output.txt", str(download_dir))

            if not download_dir:
                final_pdfs.append(None)
                continue

            # 2) REMOVE WATERMARK
            try:
                rm_input = {"input_dir": download_dir}
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_json(base, f"{per_song_name}_remove_watermark_input.json", rm_input)
                processed_dir = remove_watermark.invoke(rm_input)
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_text(base, f"{per_song_name}_remove_watermark_output.txt", str(processed_dir))
            except Exception as e:
                processed_dir = None
                _record_error(new_state, f"{per_song_name}_remove_watermark_exception", e)

            if not processed_dir:
                final_pdfs.append(None)
                continue

            # 3) UPSCALE
            try:
                up_input = {"input_dir": processed_dir}
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_json(base, f"{per_song_name}_upscale_input.json", up_input)
                upscaled_dir = upscale_images.invoke(up_input)
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_text(base, f"{per_song_name}_upscale_output.txt", str(upscaled_dir))
            except Exception as e:
                upscaled_dir = None
                _record_error(new_state, f"{per_song_name}_upscale_exception", e)

            if not upscaled_dir:
                final_pdfs.append(None)
                continue

            # Use final key/instrument if scraper adjusted them
            actual_key = SCRAPE_METADATA.get("key") or song.get("key", "")
            actual_instrument = SCRAPE_METADATA.get("instrument") or song.get("instrument", "")

            # Update SCRAPE_METADATA for downstream
            try:
                SCRAPE_METADATA["key"] = actual_key
                SCRAPE_METADATA["instrument"] = actual_instrument
            except Exception:
                pass

            safe_instrument = sanitize_title(actual_instrument or "Unknown Instrument")
            safe_key = sanitize_title(actual_key or "Unknown Key")

            actual_title_for_name = SCRAPE_METADATA.get("title") or song.get("title", "Unknown Title")
            safe_title_for_name = sanitize_title(actual_title_for_name)
            idx_str = str(idx + 1).zfill(2)
            filename = f"{idx_str}_{safe_title_for_name}_{safe_instrument}_{safe_key}.pdf"

            # 4) ASSEMBLE PDF
            try:
                as_input = {
                    "image_dir": upscaled_dir,
                    "output_pdf": filename,
                }
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_json(base, f"{per_song_name}_assemble_input.json", as_input)
                pdf_path = assemble_pdf.invoke(as_input)
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_text(base, f"{per_song_name}_assemble_output.txt", str(pdf_path))
            except Exception as e:
                pdf_path = None
                _record_error(new_state, f"{per_song_name}_assemble_exception", e)

            # Copy final PDF into output/orders/<date_folder>
            if pdf_path and isinstance(pdf_path, str):
                try:
                    orders_root = Path(os.getcwd()) / "output" / "orders" / date_folder
                    orders_root.mkdir(parents=True, exist_ok=True)
                    target_path = orders_root / filename

                    import shutil
                    shutil.copyfile(pdf_path, str(target_path))

                    if _debug_enabled(new_state):
                        base = _debug_dir_for(new_state)
                        _write_text(base, f"{per_song_name}_final_target.txt", str(target_path))
                except Exception as e:
                    _record_error(new_state, f"{per_song_name}_copy_exception", e)

            final_pdfs.append(pdf_path if isinstance(pdf_path, str) else None)

            # Cleanup temps
            try:
                import shutil
                for tmp_dir in list(TEMP_DIRS):
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass
                TEMP_DIRS.clear()
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_text(base, f"{per_song_name}_temp_cleanup.txt", "TEMP_DIRS cleared.")
            except Exception as e:
                _record_error(new_state, f"{per_song_name}_temp_cleanup_exception", e)

        except Exception as e:
            _record_error(new_state, f"{per_song_name}_outer_exception", e)
            final_pdfs.append(None)

    new_state["final_pdfs"] = final_pdfs

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _write_json(base, "final_state.json", {
            k: v for k, v in new_state.items() if k not in ("songs",)  # songs can be large; already saved
        })

    _record_timing(new_state, "process_songs_node", _stop_timer(start))
    return new_state



# =========================
# Iterative per-song nodes
# =========================

def init_loop_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize loop state before per-song processing."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    # Initialize loop counters and accumulators
    if "song_idx" not in new_state or not isinstance(new_state.get("song_idx"), int):
        new_state["song_idx"] = 0
    new_state.setdefault("final_pdfs", [])

    # Ensure derived order folder is computed once
    try:
        new_state["_date_folder"] = _get_order_folder(new_state)
    except Exception:
        new_state["_date_folder"] = "unknown_date"

    _record_timing(new_state, "init_loop_node", _stop_timer(start))
    return new_state


def _per_song_labels(new_state: Dict[str, Any]) -> tuple[int, Dict[str, Any], str, str]:
    """Utility: fetch current song payload and build nice debug name/prefix."""
    idx = int(new_state.get("song_idx", 0))
    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    song = songs.get(idx, {}) if isinstance(songs, dict) else {}
    # Build names for artifacts
    safe_title = sanitize_title(song.get("title", "Unknown Title"))
    per_song_prefix = f"song_{str(idx + 1).zfill(2)}"
    per_song_name = f"{per_song_prefix}_{safe_title}"
    return idx, song, per_song_prefix, per_song_name


def scraper_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 1: scrape; produces download_dir in state."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _ensure_file_handler(new_state, base)
        _logger.debug("scraper_node: begin")

    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)
    input_dir_override = new_state.get("input_dir", "data/samples")

    scrape_input = {
        "title": song.get("title", ""),
        "instrument": song.get("instrument", ""),
        "key": song.get("key", ""),
        "input_dir": input_dir_override,
        "artist": song.get("artist", ""),
    }
    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _write_json(base, f"{per_song_name}_scrape_input.json", scrape_input)

    download_dir = None
    try:
        download_dir = scrape_music.invoke(scrape_input)
    except Exception as e:
        _record_error(new_state, f"{per_song_name}_scrape_exception", e)

    if _debug_enabled(new_state):
        base = _debug_dir_for(new_state)
        _write_text(base, f"{per_song_name}_scrape_output.txt", str(download_dir))

    new_state["download_dir"] = download_dir
    _record_timing(new_state, "scraper_node", _stop_timer(start))
    return new_state


def watermark_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: watermark removal; consumes download_dir, produces processed_dir."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)
    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)

    download_dir = new_state.get("download_dir")
    processed_dir = None
    if download_dir:
        try:
            rm_input = {"input_dir": download_dir}
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_remove_watermark_input.json", rm_input)
            processed_dir = remove_watermark.invoke(rm_input)
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_remove_watermark_output.txt", str(processed_dir))
        except Exception as e:
            _record_error(new_state, f"{per_song_name}_remove_watermark_exception", e)

    new_state["processed_dir"] = processed_dir
    _record_timing(new_state, "watermark_node", _stop_timer(start))
    return new_state


def upscaler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: upscaling; consumes processed_dir, produces upscaled_dir."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)
    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)

    processed_dir = new_state.get("processed_dir")
    upscaled_dir = None
    if processed_dir:
        try:
            up_input = {"input_dir": processed_dir}
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_upscale_input.json", up_input)
            upscaled_dir = upscale_images.invoke(up_input)
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_upscale_output.txt", str(upscaled_dir))
        except Exception as e:
            _record_error(new_state, f"{per_song_name}_upscale_exception", e)

    new_state["upscaled_dir"] = upscaled_dir
    _record_timing(new_state, "upscaler_node", _stop_timer(start))
    return new_state


def assembler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: assemble PDF for current song; advances the index and cleans up temps."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)
    upscaled_dir = new_state.get("upscaled_dir")
    date_folder = new_state.get("_date_folder") or _get_order_folder(new_state)

    pdf_path: str | None = None
    if upscaled_dir:
        # Build filename using possibly adjusted metadata from scraper
        try:
            actual_key = SCRAPE_METADATA.get("key") or song.get("key", "")
            actual_instrument = SCRAPE_METADATA.get("instrument") or song.get("instrument", "")
            # Persist back for downstream (mirrors existing behavior)
            try:
                SCRAPE_METADATA["key"] = actual_key
                SCRAPE_METADATA["instrument"] = actual_instrument
            except Exception:
                pass

            safe_instrument = sanitize_title(actual_instrument or "Unknown Instrument")
            safe_key = sanitize_title(actual_key or "Unknown Key")
            actual_title_for_name = SCRAPE_METADATA.get("title") or song.get("title", "Unknown Title")
            safe_title_for_name = sanitize_title(actual_title_for_name)
            idx_str = str(idx + 1).zfill(2)
            filename = f"{idx_str}_{safe_title_for_name}_{safe_instrument}_{safe_key}.pdf"

            as_input = {"image_dir": upscaled_dir, "output_pdf": filename}
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_assemble_input.json", as_input)
            pdf_path = assemble_pdf.invoke(as_input)
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_assemble_output.txt", str(pdf_path))

            # Copy into orders folder
            if pdf_path and isinstance(pdf_path, str):
                try:
                    orders_root = Path(os.getcwd()) / "output" / "orders" / date_folder
                    orders_root.mkdir(parents=True, exist_ok=True)
                    target_path = orders_root / filename
                    import shutil as _shutil
                    _shutil.copyfile(pdf_path, str(target_path))
                    if _debug_enabled(new_state):
                        base = _debug_dir_for(new_state)
                        _write_text(base, f"{per_song_name}_final_target.txt", str(target_path))
                except Exception as e:
                    _record_error(new_state, f"{per_song_name}_copy_exception", e)
        except Exception as e:
            _record_error(new_state, f"{per_song_name}_assemble_outer_exception", e)

    # Record result (even if None for failures)
    final_pdfs: List[str | None] = list(new_state.get("final_pdfs", []))
    final_pdfs.append(pdf_path if isinstance(pdf_path, str) else None)
    new_state["final_pdfs"] = final_pdfs

    # Cleanup TEMP_DIRS between songs
    try:
        import shutil as _shutil
        for tmp_dir in list(TEMP_DIRS.values()):
            try:
                _shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        TEMP_DIRS.clear()
        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_text(base, f"{per_song_name}_temp_cleanup.txt", "TEMP_DIRS cleared.")
    except Exception as e:
        _record_error(new_state, f"{per_song_name}_temp_cleanup_exception", e)

    # Advance to next song
    new_state["song_idx"] = int(new_state.get("song_idx", 0)) + 1

    # If this was the last song, write a final_state snapshot for debugging
    try:
        songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
        if new_state["song_idx"] >= len(songs):
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, "final_state.json", {
                    k: v for k, v in new_state.items() if k not in ("songs",)
                })
    except Exception:
        pass

    _record_timing(new_state, "assembler_node", _stop_timer(start))
    return new_state


def should_continue(state: Dict[str, Any]) -> str:
    """Conditional router after assembling a song."""
    try:
        idx = int(state.get("song_idx", 0))
        songs: Dict[int, Dict[str, Any]] = state.get("songs", {}) or {}
        return "continue" if idx < len(songs) else "end"
    except Exception:
        return "end"


def compile_graph() -> Any:
    """Construct and compile the order-of-worship graph with per-song loop over four steps."""
    graph = StateGraph(dict)
    # Parse instruction and extract songs (unchanged)
    graph.add_node("parser", parser_node)
    graph.add_node("extractor", extract_songs_node)

    # New: four explicit per-song steps + init
    graph.add_node("init", init_loop_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("watermark", watermark_node)
    graph.add_node("upscaler", upscaler_node)
    graph.add_node("assembler", assembler_node)

    # Wiring
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "extractor")
    graph.add_edge("extractor", "init")
    graph.add_edge("init", "scraper")
    graph.add_edge("scraper", "watermark")
    graph.add_edge("watermark", "upscaler")
    graph.add_edge("upscaler", "assembler")

    # Loop: after assembling, either continue with next song or end
    graph.add_conditional_edges(
        "assembler",
        should_continue,
        {
            "continue": "scraper",
            "end": END,
        },
    )

    return graph.compile()

# Expose compiled graph
graph = compile_graph()
