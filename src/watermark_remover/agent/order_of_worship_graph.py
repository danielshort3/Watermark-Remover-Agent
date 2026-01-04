"""LangGraph pipeline for processing an order of worship PDF with extensive debugging
and LLM-only natural-language handling (no heuristic NLP).

Key changes:
- Adds robust debugging (run IDs, per-node timings, structured errors, persisted artifacts).
- Delegates parsing to run_instruction (agent-backed, with direct-LLM fallback) to extract song metadata.
- Removes all regex/rule-based NLP parsing for instructions and songs.

Debug artifacts are written to:
  output/logs/<RUN_TS>/orders/<order_folder or 'unknown_date'>/<run_id>/
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

# Note: this module was moved under the src/ layout for a cleaner structure.

import re
import os
import json
import time
import uuid
import traceback
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path



# --- Runtime safety knobs ----------------------------------------------------
# Make recursion errors less likely in deep tool/graph stacks.
try:
    import sys as _sys
    _RL = int(os.getenv("ORDER_RECURSION_LIMIT", "10000"))
    if _RL > _sys.getrecursionlimit():
        _sys.setrecursionlimit(_RL)
except Exception:
    # If setting fails (e.g., restricted runtime), just continue.
    pass
# -----------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor as _PPExecutor, as_completed as _as_completed

# Import tools and utilities from your project
from watermark_remover.agent.tools import sanitize_title, SCRAPE_METADATA, TEMP_DIRS
from utils.transposition_utils import normalize_key


def _iter_temp_dirs():
    """Return a *list* of temp directories regardless of TEMP_DIRS' type.

    Supports both list[str] and dict[str, str] (legacy). Ignores non-str items.
    """
    try:
        td = TEMP_DIRS
        if isinstance(td, dict):
            it = list(td.values())
        else:
            it = list(td or [])
    except Exception:
        it = []
    return [p for p in it if isinstance(p, str)]
from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
    SCRAPE_METADATA,
    remove_watermark_batch,
    upscale_images_batch,
    get_log_root,
    apply_concise_debug_filter,
)
try:
    from watermark_remover.agent.prompts import (
        build_order_parser_prompt,
        build_song_extractor_prompt,
        log_prompt,
    )
except Exception:
    build_order_parser_prompt = None  # type: ignore
    build_song_extractor_prompt = None  # type: ignore

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
_PDFMINER_LOGGERS = (
    "pdfminer",
    "pdfminer.psparser",
    "pdfminer.pdfinterp",
    "pdfminer.pdfdevice",
    "pdfminer.pdfpage",
)

def _env_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() not in ("", "0", "false", "no", "off")

def _debug_enabled(state: Dict[str, Any]) -> bool:
    if isinstance(state.get("debug"), bool):
        return state["debug"]
    return _env_truthy(os.getenv("ORDER_DEBUG", "1"))


def _cancel_requested(state: Dict[str, Any]) -> bool:
    event = state.get("_cancel_event")
    if event is None or not hasattr(event, "is_set"):
        return False
    try:
        return bool(event.is_set())
    except Exception:
        return False


def _mark_cancelled(state: Dict[str, Any], label: str) -> None:
    state["cancelled"] = True
    try:
        _record_error(state, f"{label}.cancelled", RuntimeError("Cancelled by user."))
    except Exception:
        pass

def _ensure_logger_configured(level: int) -> None:
    # Avoid duplicate handlers if module reloaded
    if not _logger.handlers:
        _logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        apply_concise_debug_filter(ch)
        _logger.addHandler(ch)
    else:
        for handler in _logger.handlers:
            apply_concise_debug_filter(handler)
    # pdfminer can emit extremely verbose debug logs; clamp them to WARNING.
    for name in _PDFMINER_LOGGERS:
        plogger = logging.getLogger(name)
        plogger.setLevel(logging.WARNING)

def _input_root(state: Dict[str, Any]) -> str:
    """Return an absolute path to the input directory that stores order PDFs."""
    root_dir = (state.get("pdf_root") or state.get("input_dir") or "input") or "input"
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.getcwd(), root_dir)
    return root_dir

def _display_input_root(state: Dict[str, Any]) -> str:
    """Return a human-friendly label for the input directory."""
    try:
        root = _input_root(state)
    except Exception:
        return "input"
    try:
        rel = os.path.relpath(root, os.getcwd())
        if not rel.startswith(".."):
            return rel
    except Exception:
        pass
    return root

def _tokenize_for_match(value: str) -> list[str]:
    """Split a string into lowercase tokens (digits stripped of leading zeros)."""
    tokens: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", value.lower()):
        if not part:
            continue
        if part.isdigit():
            try:
                part = str(int(part))
            except Exception:
                part = part.lstrip("0") or "0"
        tokens.append(part)
    return tokens

def _normalize_pdf_key(value: str) -> str:
    """Return a simplified comparison key for filenames/instructions."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_extracted_key(value: str) -> str:
    """Normalize extracted keys, preferring the starting key in modulations."""
    raw = (value or "").strip()
    if not raw:
        return ""
    match = re.search(r"([A-Ga-g](?:#|b)?)\s*[-–—]\s*([A-Ga-g](?:#|b)?)", raw)
    if match:
        normalized = normalize_key(match.group(1))
        return normalized or raw
    return raw

def _normalized_dates_from_string(value: str) -> set[str]:
    """Extract normalized MM_DD_YYYY strings from arbitrary text."""
    results: set[str] = set()
    for _, _, y, m, d in _try_parse_date_fragment(value):
        results.add(_normalize_date_mm_dd_yyyy(y, m, d))
    return results

def _list_pdf_candidates(state: Dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate PDFs under the configured input directory with match metadata."""
    try:
        root = _input_root(state)
    except Exception:
        return []
    root_path = Path(root)
    if not root_path.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for entry in sorted(root_path.glob("*.pdf")):
        name = entry.name
        tokens = set(_tokenize_for_match(name))
        candidates.append(
            {
                "path": str(entry),
                "name": name,
                "normalized": _normalize_pdf_key(name),
                "tokens": tokens,
                "dates": _normalized_dates_from_string(name),
            }
        )
    return candidates

def _format_available_pdfs_for_prompt(
    candidates: list[dict[str, Any]],
    root_label: str,
) -> str:
    """Return a human-readable list of available PDFs for LLM context."""
    if not candidates:
        return f"\n\nAvailable order PDFs under {root_label}:\n- (none found)"
    max_list = 20
    lines = [f"- {c['name']}" for c in candidates[:max_list]]
    extra = len(candidates) - max_list
    if extra > 0:
        lines.append(f"- ... (+{extra} more)")
    listing = "\n".join(lines)
    return f"\n\nAvailable order PDFs under {root_label}:\n{listing}"

def _build_pdf_query_info(text: str) -> dict[str, Any]:
    tokens = set(_tokenize_for_match(text))
    return {
        "raw": text,
        "tokens": tokens,
        "normalized": _normalize_pdf_key(text),
        "dates": _normalized_dates_from_string(text),
    }

def _score_candidate_against_queries(
    candidate: dict[str, Any],
    queries: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    tokens: set[str] = candidate.get("tokens") or set()
    dates: set[str] = candidate.get("dates") or set()
    normalized = candidate.get("normalized") or ""
    score = 0.0
    matched_tokens: set[str] = set()
    matched_dates: set[str] = set()
    normalized_hit = False

    for query in queries:
        qt = query.get("tokens") or set()
        intersection = tokens & qt
        if intersection:
            matched_tokens |= intersection
            score += len(intersection)
        qd = query.get("dates") or set()
        date_overlap = dates & qd
        if date_overlap:
            matched_dates |= date_overlap
            score += len(date_overlap) * 10
        qn = query.get("normalized") or ""
        if qn and qn == normalized:
            score += 25
            normalized_hit = True
        elif qn and normalized and (qn in normalized or normalized in qn):
            score += 5

    detail = {
        "matched_tokens": sorted(matched_tokens),
        "matched_dates": sorted(matched_dates),
        "normalized_hit": normalized_hit,
    }
    return score, detail

def _resolve_pdf_from_instruction(
    state: Dict[str, Any],
    desired_pdf: str,
    instruction: str,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Map the instruction/desired name to an actual PDF path if possible."""
    if candidates is None:
        candidates = _list_pdf_candidates(state)
    if not candidates:
        return None, {"reason": "no_candidates"}

    queries: list[dict[str, Any]] = []
    desired_pdf = (desired_pdf or "").strip()
    instruction = (instruction or "").strip()
    if desired_pdf:
        queries.append(_build_pdf_query_info(desired_pdf))
    if instruction:
        queries.append(_build_pdf_query_info(instruction))
    if not queries:
        return None, {"reason": "no_queries"}

    combined_dates: set[str] = set()
    for q in queries:
        combined_dates |= q.get("dates") or set()

    best_idx = -1
    best_score = 0.0
    best_detail: dict[str, Any] | None = None
    for idx, cand in enumerate(candidates):
        score, detail = _score_candidate_against_queries(cand, queries)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_detail = detail

    if best_score <= 0 and combined_dates:
        for idx, cand in enumerate(candidates):
            pdf_date = cand.get("pdf_date")
            if pdf_date is None:
                try:
                    pdf_date = _determine_order_folder_from_pdf(state, cand["path"])
                except Exception as exc:  # pragma: no cover - diagnostics only
                    _record_error(state, "pdf_resolution.date_probe_exception", exc)
                    pdf_date = None
                cand["pdf_date"] = pdf_date
            if pdf_date and pdf_date in combined_dates:
                best_idx = idx
                best_score = 100.0
                best_detail = {"matched_pdf_date": pdf_date}
                break

    if best_idx == -1 or best_detail is None:
        return None, {
            "reason": "no_match",
            "candidates": [c["name"] for c in candidates],
        }

    candidate = candidates[best_idx]
    best_detail = dict(best_detail)
    best_detail["score"] = best_score
    best_detail["resolved_name"] = candidate["name"]
    best_detail["path"] = candidate["path"]
    return candidate["path"], best_detail

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
    hook = state.get("_debug_order_folder")
    order_folder = hook if isinstance(hook, str) and hook else _get_order_folder(state)
    log_root = Path(get_log_root(create=True))
    base = log_root / "orders" / order_folder / run_id
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


def _append_error(state: Dict[str, Any], label: str, message: str) -> None:
    errs: List[Dict[str, Any]] = state.get("errors", [])
    info = {
        "label": label,
        "type": "RuntimeError",
        "message": message,
        "traceback": "",
    }
    errs.append(info)
    state["errors"] = errs
    if _debug_enabled(state):
        base = _debug_dir_for(state)
        _write_json(base, "errors.json", errs)


def _add_jobs_from_downloads(
    state: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    final_pdfs: List[str | None],
    *,
    idx: int,
    song: Dict[str, Any],
    per_song_name: str,
    downloads: List[Dict[str, Any]],
    debug_enabled: bool,
    debug_base: Path | None,
) -> None:
    if not downloads:
        title = str(song.get("title") or "")
        msg = f"No song candidates found for '{title}'." if title else "No song candidates found."
        _append_error(state, f"{per_song_name}_no_candidates", msg)
        final_pdfs.append(None)
        return

    for match_idx, dl in enumerate(downloads, 1):
        image_dir = dl.get("image_dir")
        existing_pdf = dl.get("existing_pdf")
        if not image_dir and not existing_pdf:
            final_pdfs.append(None)
            continue

        variant_suffix = f"_match{match_idx}"
        result_index = len(final_pdfs)
        final_pdfs.append(None)

        meta = dl.get("meta") or {}
        actual_key = meta.get("key") or song.get("key", "")
        actual_instrument = meta.get("instrument") or song.get("instrument", "")
        actual_title_for_name = meta.get("title") or song.get("title", "Unknown Title")
        safe_instrument = sanitize_title(actual_instrument or "Unknown Instrument")
        safe_key = sanitize_title(actual_key or "Unknown Key")
        safe_title_for_name = sanitize_title(actual_title_for_name)
        idx_str = str(idx + 1).zfill(2)
        rank_str = str(match_idx).zfill(2)
        filename = f"{idx_str}_{rank_str}_{safe_title_for_name}_{safe_instrument}_{safe_key}.pdf"

        job: Dict[str, Any] = {
            "song_index": idx,
            "song": song,
            "meta": meta,
            "result_index": result_index,
            "per_song_name": per_song_name,
            "variant_suffix": variant_suffix,
            "filename": filename,
            "assembly_meta": {
                "title": actual_title_for_name,
                "artist": meta.get("artist", ""),
                "instrument": actual_instrument,
                "key": actual_key,
            },
            "existing_pdf": existing_pdf,
            "download_dir": image_dir,
            "abs_download_dir": os.path.abspath(image_dir) if image_dir else None,
            "rank_str": rank_str,
        }
        jobs.append(job)

        if debug_enabled and debug_base is not None and image_dir:
            rm_input = {"input_dir": image_dir}
            _write_json(debug_base, f"{per_song_name}{variant_suffix}_remove_watermark_input.json", rm_input)

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

MAX_LLM_RETRIES = int(os.getenv("ORDER_MAX_LLM_RETRIES", "1"))

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
        prompt_path = None
        if base is not None:
            prompt_path = _write_text(base, f"{label}_prompt.attempt{attempt}.txt", this_prompt)

        # Call LLM
        raw = None
        err_msg = None
        parsed: Any = None
        prev_label = os.environ.get("WMRA_LLM_TRACE_LABEL")
        prev_prompt = os.environ.get("WMRA_LLM_TRACE_PROMPT_PATH")
        try:
            os.environ["WMRA_LLM_TRACE_LABEL"] = attempt_label
            if prompt_path:
                os.environ["WMRA_LLM_TRACE_PROMPT_PATH"] = prompt_path
            raw = run_instruction(this_prompt)
            # Also log centrally under output/logs/<RUN_TS>/prompts
            try:
                from watermark_remover.agent.prompts import log_prompt, log_llm_response  # type: ignore
                log_prompt(f"{label}_attempt{attempt}", this_prompt)
                log_llm_response(f"{label}_attempt{attempt}", raw)
            except Exception:
                pass
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
        finally:
            if prev_label is None:
                os.environ.pop("WMRA_LLM_TRACE_LABEL", None)
            else:
                os.environ["WMRA_LLM_TRACE_LABEL"] = prev_label
            if prev_prompt is None:
                os.environ.pop("WMRA_LLM_TRACE_PROMPT_PATH", None)
            else:
                os.environ["WMRA_LLM_TRACE_PROMPT_PATH"] = prev_prompt

    # Persist attempts summary
    if base is not None:
        _write_json(base, f"{label}_attempts.json", attempts_log)

    return parsed


# ---------------------------------------------------------------------------
# PDF Text Extraction (not NLP; just I/O)
# ---------------------------------------------------------------------------

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    from pdfminer.pdfpage import PDFPage  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore
    PDFPage = None  # type: ignore

def _read_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF using pdfminer.six with verbose tracing."""
    _logger.info("Attempting to extract text from PDF via pdfminer: %s", pdf_path)
    if pdfminer_extract_text is None or PDFPage is None:
        msg = "pdfminer.six is unavailable. Install pdfminer.six to enable PDF parsing."
        _logger.error(msg)
        raise RuntimeError(msg)

    if not os.path.exists(pdf_path):
        msg = f"PDF path not found: {pdf_path}"
        _logger.error(msg)
        raise RuntimeError(msg)

    try:
        page_count = 0
        with open(pdf_path, "rb") as fh:
            for page_count, _ in enumerate(PDFPage.get_pages(fh), start=1):
                _logger.info("Detected PDF page %d", page_count)
        _logger.info("pdfminer detected %d total pages", page_count)
        if page_count == 0:
            _logger.warning("pdfminer did not detect any pages in %s", pdf_path)

        _logger.info("Running pdfminer text extraction.")
        text = pdfminer_extract_text(pdf_path) or ""
        _logger.info("Completed pdfminer extraction; text length: %d characters", len(text))
        if not text.strip():
            _logger.warning("pdfminer returned empty text for %s", pdf_path)
        return text
    except Exception as exc:
        _logger.exception("pdfminer failed during PDF processing: %s", pdf_path)
        raise RuntimeError("pdfminer could not read the PDF.") from exc


def _get_pdf_text_cached(state: Dict[str, Any], pdf_path: str) -> str:
    """Return cached PDF text for a path, reading from disk only once."""
    cache = state.get("_pdf_text_cache")
    if not isinstance(cache, dict):
        cache = {}
        state["_pdf_text_cache"] = cache
    cached = cache.get(pdf_path)
    if isinstance(cached, str):
        return cached
    text = _read_pdf_text(pdf_path)
    cache[pdf_path] = text
    return text


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


def _build_song_and_date_extractor_prompt(pdf_text: str, user_req: str | None) -> str:
    """Prompt to extract both date and songs in one pass."""
    rules = (
        "You are a precise parser for 'order of worship' text. "
        "Return STRICT JSON with exactly two keys: "
        "date (string, MM_DD_YYYY or empty), "
        "songs (object mapping zero-based indices to {title, artist, key}). "
        "If details cannot be found for a field, use an empty string. "
        "Do not fabricate songs not present in the text."
    )
    date_rules = (
        "\n\nDATE RULES:\n"
        "- Prefer dates near phrases like 'Date of Service', 'Service Date', "
        "'Order of Worship for', or weekday names.\n"
        "- Return MM_DD_YYYY (zero-padded) or an empty string if missing.\n"
    )
    song_rules = (
        "\n\nSONG DETECTION RULES:\n"
        "- Treat any line of the form '<Title> [ <Artist> in <Key> ]' as a SONG entry.\n"
        "- Allow a leading timestamp (e.g., '3:25' or '4:56') before the title; still a SONG.\n"
        "- Include titles even if they look like generic labels (e.g., 'Praise', 'Response'):\n"
        "  if they appear with '[ <Artist> in <Key> ]', they are SONGS.\n"
        "- Ignore obvious non-songs without '[ Artist in Key ]' like 'Pre Service', 'Announcements',\n"
        "  'Offering Prayer', 'Sermon', 'Scripture', 'Rehearsal Times', or 'Length in mins'.\n"
        "- Keep duplicate titles if they appear in different keys or at different times.\n"
        "- Accept 'Default Arrangement' as a valid artist.\n"
        "- Example: '4:56 Praise [ Elevation Worship in E ]' -> title='Praise', artist='Elevation Worship', key='E'.\n"
        "- If a key is a modulation like 'F-G', return only the starting key (e.g., 'F').\n"
    )
    selection_rules = (
        "\n\nSELECTION RULES (apply strictly if present):\n"
        "- Respect user instructions like 'do not download <title>' or 'only download the fourth song'.\n"
        "- Interpret ordinals in the order of appearance.\n"
        "- If indices are provided, treat them as 1-based unless clearly zero-based.\n"
        "- Exclusions first, then inclusions.\n"
        "- Never fabricate songs; only pick from RAW TEXT.\n"
    )
    user_section = f"\n\nUSER REQUESTS:\n{user_req}" if user_req else ""
    return (
        rules
        + date_rules
        + song_rules
        + selection_rules
        + user_section
        + f"\n\nRAW ORDER OF WORSHIP TEXT:\n{pdf_text}\n\nReturn ONLY JSON."
    )

def _extract_service_date_with_llm(state: Dict[str, Any], pdf_text: str) -> str | None:
    if run_instruction is None:
        return None
    try:
        prompt = _build_date_extractor_prompt(pdf_text)
        try:
            from watermark_remover.agent.prompts import log_prompt
            log_prompt("date_extractor", prompt)
        except Exception:
            pass
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
    """Read PDF text (cached) and use heuristics to produce MM_DD_YYYY, else 'unknown'."""
    _logger.info("Determining order folder from PDF: %s", pdf_path)
    try:
        raw = _get_pdf_text_cached(state, pdf_path)
    except Exception as e:
        _logger.exception("Failed to read PDF text for date extraction: %s", pdf_path)
        _record_error(state, "date_extractor.read_pdf_exception", e)
        raw = ""

    if _debug_enabled(state):
        state["_debug_order_folder"] = state.get("order_folder") or "unknown"
        base = _debug_dir_for(state)
        _write_text(base, "pdf_text.txt", raw)

    if raw:
        _logger.info("PDF text extracted; length=%d. Proceeding to heuristic date extraction.", len(raw))
    else:
        _logger.warning("No text available from PDF; heuristic date extraction may fail.")

    date = _extract_service_date_heuristic(raw)

    if not date:
        _logger.warning("Unable to extract service date; defaulting to 'unknown'.")
        date = "unknown"
    else:
        _logger.info("Extracted service date candidate: %s", date)

    # Persist decision for debugging
    if _debug_enabled(state):
        state["_debug_order_folder"] = state.get("order_folder") or date
        base = _debug_dir_for(state)
        _write_text(base, "date_of_service.txt", date)

    # Sanitize folder name
    try:
        return sanitize_title(date)
    except Exception:
        return date or "unknown"


def _ensure_order_pdf_copied(state: Dict[str, Any]) -> None:
    """Copy the source order-of-worship PDF into output/orders/<date>/
    with filename pattern: 00_<Month>_<DD>_<YYYY>_Order_Of_Worship.pdf.

    Example: for 08_31_2025 the file becomes
    00_August_31_2025_Order_Of_Worship.pdf
    """
    try:
        pdf_name = (state.get("pdf_name") or "").strip()
        if not pdf_name:
            return

        # Resolve path similar to other utilities
        pdf_path = pdf_name
        if not os.path.isabs(pdf_path):
            root_dir = state.get("pdf_root") or state.get("input_dir") or "input"
            candidate = os.path.join(os.getcwd(), root_dir, pdf_path)
            pdf_path = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), pdf_path)
        if not os.path.exists(pdf_path):
            return

        date_folder = state.get("_date_folder") or _get_order_folder(state) or "unknown_date"

        orders_root = Path(os.getcwd()) / "output" / "orders" / date_folder
        orders_root.mkdir(parents=True, exist_ok=True)
        # Build pretty filename using Month name if date_folder is MM_DD_YYYY
        try:
            import re as _re
            import calendar as _cal
            m = _re.fullmatch(r"(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<y>\d{4})", date_folder)
            if m:
                month_num = int(m.group("m"))
                day_num = int(m.group("d"))
                year_num = int(m.group("y"))
                month_name = _cal.month_name[month_num] if 1 <= month_num <= 12 else str(month_num)
                pretty = f"{month_name}_{day_num:02d}_{year_num:04d}"
            else:
                # Fallback: just use the folder name
                pretty = date_folder
        except Exception:
            pretty = date_folder

        target = orders_root / f"00_{pretty}_Order_Of_Worship.pdf"

        if not target.exists():
            import shutil as _shutil
            _shutil.copyfile(pdf_path, str(target))

        if _debug_enabled(state):
            try:
                base = _debug_dir_for(state)
                _write_text(base, "order_pdf_target.txt", str(target))
            except Exception:
                pass

    except Exception as e:
        _record_error(state, "ensure_order_pdf_copied.exception", e)


# ---------------------------------------------------------------------------
# Parallel Scrape Worker (separate process + browser)
# ---------------------------------------------------------------------------

def _scrape_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape a single song in an isolated process and return downloads."""
    import os as _os
    import traceback as _tb
    from typing import Any as _Any, Dict as _Dict, List as _List

    errors: _List[str] = []
    idx: int = int(payload.get("idx", 0))
    song: _Dict[str, _Any] = dict(payload.get("song") or {})
    input_dir: str = str(payload.get("input_dir") or "data/samples")
    top_n: int = int(payload.get("top_n", 3) or 3)
    log_dir: str = str(payload.get("log_dir") or "")
    run_ts: str = str(payload.get("run_ts") or "")
    preview_root: str = str(payload.get("preview_root") or "")

    try:
        from watermark_remover.agent.tools import (
            scrape_music as _scrape,
            SCRAPE_METADATA as _META,
            init_pipeline_logging as _init_logging,
        )
    except Exception as e:  # pragma: no cover
        errors.append(f"imports: {e}")
        return {"idx": idx, "downloads": [], "errors": errors}

    if run_ts:
        _os.environ["RUN_TS"] = run_ts
    if log_dir:
        _os.environ["WMRA_LOG_DIR"] = log_dir
    if preview_root:
        try:
            preview_dir = _os.path.join(preview_root, f"worker_{_os.getpid()}")
            _os.makedirs(preview_dir, exist_ok=True)
            _os.environ["WMRA_PREVIEW_DIR"] = preview_dir
        except Exception:
            _os.environ.pop("WMRA_PREVIEW_DIR", None)
    try:
        _init_logging()
    except Exception:
        pass

    downloads: _List[_Dict[str, _Any]] = []
    try:
        raw = _scrape.invoke({
            "title": song.get("title", ""),
            "instrument": song.get("instrument", ""),
            "key": song.get("key", ""),
            "artist": song.get("artist", ""),
            "input_dir": input_dir,
            "top_n": top_n,
        })
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    if it.get("image_dir") or it.get("existing_pdf"):
                        downloads.append({
                            "image_dir": it.get("image_dir"),
                            "existing_pdf": it.get("existing_pdf"),
                            "meta": it.get("meta", {}),
                        })
        elif isinstance(raw, str):
            downloads.append({
                "image_dir": raw,
                "meta": {
                    "title": _META.get("title", song.get("title", "")),
                    "artist": _META.get("artist", song.get("artist", "")),
                    "instrument": _META.get("instrument", song.get("instrument", "")),
                    "key": _META.get("key", song.get("key", "")),
                },
            })
    except Exception as e:  # pragma: no cover
        errors.append(f"scrape: {e}\n{_tb.format_exc()}")

    return {"idx": idx, "downloads": downloads, "errors": errors}


# ---------------------------------------------------------------------------
# Parallel Per‑Song Worker (separate process + browser)
# ---------------------------------------------------------------------------

def _song_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single song pipeline in an isolated process.

    This function is designed to be executed inside a child process. It
    sets a per‑song RUN_TS/WMRA_LOG_DIR, invokes the scrape → watermark →
    upscale → assemble steps, and copies the final PDF(s) into
    output/orders/<date>/ using the index+rank naming convention.

    Parameters
    ----------
    payload: Dict[str, Any]
        A dict containing:
        - idx: int (0‑based song index)
        - song: {title, instrument, key, artist?}
        - input_dir: str (library override)
        - date_folder: str (orders/<date> folder)
        - top_n: int (number of candidates to consider)
        - run_id: str (base run id; used to derive per‑song RUN_TS)

    Returns
    -------
    Dict[str, Any]
        {"idx": int, "pdfs": list[str|None], "errors": list[str]}
    """
    # Imports inside the worker keep parent load light and avoid import‑time side‑effects.
    import os as _os
    import traceback as _tb
    import json as _json
    import time as _time
    from pathlib import Path as _Path
    from typing import Any as _Any, Dict as _Dict, List as _List

    errors: _List[str] = []
    idx: int = int(payload.get("idx", 0))
    song: _Dict[str, _Any] = dict(payload.get("song") or {})
    input_dir: str = str(payload.get("input_dir") or "data/samples")
    date_folder: str = str(payload.get("date_folder") or "unknown_date")
    top_n: int = int(payload.get("top_n", 3) or 3)
    run_id: str = str(payload.get("run_id") or "run")

    # Lazy imports of tools to ensure they are resolved in child process.
    try:
        from watermark_remover.agent.tools import (
            scrape_music as _scrape,
            remove_watermark as _rm,
            upscale_images as _up,
            assemble_pdf as _asm,
            SCRAPE_METADATA as _META,
            TEMP_DIRS as _TEMPS,
            sanitize_title as _sanitize,
            init_pipeline_logging as _init_logging,
        )
    except Exception as e:  # pragma: no cover - only in runtime, not tests
        errors.append(f"imports: {e}")
        return {"idx": idx, "pdfs": [], "errors": errors}

    # Normalise a base name for debug artifacts.
    safe_title = _sanitize(song.get("title", "Unknown Title"))
    per_song_prefix = f"song_{str(idx + 1).zfill(2)}"
    per_song_name = f"{per_song_prefix}_{safe_title}"

    # Configure a per‑song log directory nested under the order's debug root.
    # This unifies all artifacts (graph + tools) under one location.
    run_ts = f"{run_id}_song{idx+1:02d}"
    _os.environ["RUN_TS"] = run_ts
    base_root = _Path(get_log_root(create=True)) / "orders" / date_folder / run_id / "songs" / per_song_name
    base_root.mkdir(parents=True, exist_ok=True)
    _os.environ["WMRA_LOG_DIR"] = str(base_root)
    # Record start time for diagnostics
    try:
        (base_root / "worker_times.json").write_text(_json.dumps({"start": _time.time()}), encoding="utf-8")
    except Exception:
        pass

    # Ensure pipeline log handlers (per-process)
    try:
        _init_logging()
    except Exception:
        pass

    # 1) Scrape
    step_times: dict[str, float] = {}
    _t0 = _time.time()
    downloads: list[dict] = []
    try:
        raw = _scrape.invoke({
            "title": song.get("title", ""),
            "instrument": song.get("instrument", ""),
            "key": song.get("key", ""),
            "artist": song.get("artist", ""),
            "input_dir": input_dir,
            "top_n": top_n,
        })
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    if it.get("image_dir") or it.get("existing_pdf"):
                        downloads.append({
                            "image_dir": it.get("image_dir"),
                            "existing_pdf": it.get("existing_pdf"),
                            "meta": it.get("meta", {}),
                        })
        elif isinstance(raw, str):
            downloads.append({
                "image_dir": raw,
                "meta": {
                    "title": _META.get("title", song.get("title", "")),
                    "artist": _META.get("artist", song.get("artist", "")),
                    "instrument": _META.get("instrument", song.get("instrument", "")),
                    "key": _META.get("key", song.get("key", "")),
                },
            })
    except Exception as e:  # pragma: no cover
        errors.append(f"scrape: {e}\n{_tb.format_exc()}")
    step_times["scrape"] = _time.time() - _t0


    # Early exit if nothing to process
    if not downloads:
        title = str(song.get("title") or "")
        message = f"No song candidates found for '{title}'." if title else "No song candidates found."
        errors.append(message)
        return {"idx": idx, "pdfs": [], "errors": errors}

    # 2) Remove watermark, 3) Upscale
    for dl in downloads:
        img_dir = dl.get("image_dir")
        if not img_dir:
            continue
        try:
            _t1 = _time.time()
            processed = _rm.invoke({"input_dir": img_dir})
            step_times.setdefault("watermark", 0.0)
            step_times["watermark"] += (_time.time() - _t1)
        except Exception as e:  # pragma: no cover
            errors.append(f"remove_watermark: {e}")
            processed = None
        if not processed:
            continue
        try:
            _t2 = _time.time()
            upscaled = _up.invoke({"input_dir": processed})
            step_times.setdefault("upscale", 0.0)
            step_times["upscale"] += (_time.time() - _t2)
        except Exception as e:  # pragma: no cover
            errors.append(f"upscale: {e}")
            upscaled = None
        dl["upscaled_dir"] = upscaled

    # 4) Assemble and copy into orders/<date>
    pdfs: list[str | None] = []
    for rank, dl in enumerate(downloads, 1):
        pdf_path: str | None = None
        upscaled_dir = dl.get("upscaled_dir")
        existing_pdf = dl.get("existing_pdf")
        if upscaled_dir:
            m = dict(dl.get("meta") or {})
            actual_key = m.get("key") or _META.get("key") or song.get("key", "")
            actual_instrument = m.get("instrument") or _META.get("instrument") or song.get("instrument", "")
            actual_title_for_name = m.get("title") or _META.get("title") or song.get("title", "Unknown Title")
            safe_instrument = _sanitize(actual_instrument or "Unknown Instrument")
            safe_key = _sanitize(actual_key or "Unknown Key")
            safe_title_for_name = _sanitize(actual_title_for_name)
            idx_str = str(idx + 1).zfill(2)
            rank_str = str(rank).zfill(2)
            filename = f"{idx_str}_{rank_str}_{safe_title_for_name}_{safe_instrument}_{safe_key}.pdf"
            try:
                # Keep metadata coherent for assemble step
                _META.update({
                    "title": actual_title_for_name,
                    "artist": m.get("artist", "") or _META.get("artist", ""),
                    "instrument": actual_instrument,
                    "key": actual_key,
                })
            except Exception:
                pass
            try:
                _t3 = _time.time()
                pdf_path = _asm.invoke({"image_dir": upscaled_dir, "output_pdf": filename, "meta": {
                    "title": actual_title_for_name,
                    "artist": m.get("artist", ""),
                    "instrument": actual_instrument,
                    "key": actual_key,
                }})
                step_times.setdefault("assemble", 0.0)
                step_times["assemble"] += (_time.time() - _t3)
            except Exception as e:  # pragma: no cover
                errors.append(f"assemble: {e}")
                pdf_path = None
            # Copy to orders folder
            if pdf_path and isinstance(pdf_path, str):
                try:
                    orders_root = _Path(_os.getcwd()) / "output" / "orders" / date_folder
                    orders_root.mkdir(parents=True, exist_ok=True)
                    target_path = orders_root / filename
                    import shutil as _shutil
                    _shutil.copyfile(pdf_path, str(target_path))
                    pdf_path = str(target_path)
                except Exception as e:  # pragma: no cover
                    errors.append(f"copy: {e}")
        elif existing_pdf:
            # If scraping produced an existing PDF reference, relay it (rare)
            pdf_path = str(existing_pdf)
        pdfs.append(pdf_path if isinstance(pdf_path, str) else None)

    # Best‑effort temp cleanup for this process
    try:  # pragma: no cover
        import shutil as _shutil
        for t in list(_TEMPS):
            try:
                _shutil.rmtree(t, ignore_errors=True)
            except Exception:
                pass
        _TEMPS.clear()
    except Exception:
        pass

    # Record end time
    try:
        f = base_root / "worker_times.json"
        try:
            data = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            data = {}
        data["end"] = _time.time()
        data["steps"] = step_times
        f.write_text(_json.dumps(data), encoding="utf-8")
    except Exception:
        pass

    return {"idx": idx, "pdfs": pdfs, "errors": errors}


def _terminate_executor_processes(executor: _PPExecutor) -> None:
    procs = getattr(executor, "_processes", None)
    if not isinstance(procs, dict):
        return
    for proc in procs.values():
        try:
            proc.terminate()
        except Exception:
            pass

# Prompts moved to centralized module: watermark_remover.agent.prompts

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

    available_candidates = _list_pdf_candidates(new_state)
    new_state["_available_order_pdfs"] = [c["name"] for c in available_candidates]
    available_context = _format_available_pdfs_for_prompt(available_candidates, _display_input_root(new_state))

    try:
        instruction_for_prompt = instruction + available_context
        prompt = build_order_parser_prompt(instruction_for_prompt) if build_order_parser_prompt else (
            "You are a precise data extraction assistant. Given an instruction about processing an 'order of worship' PDF, "
            "return a strict JSON with pdf_name, default_instrument, overrides, order_folder."
            f"\n\nInstruction:\n{instruction}\n{available_context}\n\nReturn ONLY the JSON object."
        )
        try:
            if 'order' in instruction.lower():
                pass
            log_prompt("order_parser", prompt)
        except Exception:
            pass

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

        resolved_pdf_path: str | None = None
        resolution_meta: dict[str, Any] | None = None
        try:
            resolved_pdf_path, resolution_meta = _resolve_pdf_from_instruction(
                new_state,
                pdf_name,
                instruction,
                available_candidates,
            )
        except Exception as exc:
            _record_error(new_state, "parser_node.pdf_resolution_exception", exc)
            resolved_pdf_path = None

        final_pdf_name = resolved_pdf_path or pdf_name

        order_folder_computed = (new_state.get("order_folder") or "").strip() or "unknown"
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

        if final_pdf_name:
            new_state["pdf_name"] = final_pdf_name

        if resolution_meta:
            meta = dict(resolution_meta)
            meta.setdefault("available_candidates", [c["name"] for c in available_candidates])
            new_state["_pdf_resolution"] = meta

        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "available_pdfs.json", [c["name"] for c in available_candidates])
            if resolution_meta:
                _write_json(base, "pdf_resolution.json", resolution_meta)
            _write_json(base, "parser_output.json", {
                "pdf_name": final_pdf_name,
                "default_instrument": default_instrument,
                "overrides": overrides,
                "order_folder": new_state.get("order_folder")
            })

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

    if run_instruction is None:
        err = RuntimeError("LLM 'run_instruction' is unavailable; cannot extract songs from PDF text.")
        _record_error(new_state, "extract_songs_node.run_instruction_missing", err)
        new_state["songs"] = {}
        _record_timing(new_state, "extract_songs_node", _stop_timer(start))
        return new_state

    try:
        # Read raw PDF text and persist it for troubleshooting
        pdf_text = _get_pdf_text_cached(new_state, pdf_path)
        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_text(base, "pdf_text.txt", pdf_text)

        # LLM prompt to extract date + songs (JSON only), honoring any user constraints in user_input
        user_req = (new_state.get("user_input") or "").strip()
        prompt = _build_song_and_date_extractor_prompt(pdf_text, user_req)
        try:
            log_prompt("song_and_date_extractor", prompt)
        except Exception:
            pass

        data = _run_llm_strict_json(
            state=new_state,
            label="song_and_date_extractor",
            prompt=prompt,
            expected="object",
        )

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON type from song extractor: {type(data).__name__}")

        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_json(base, "songs_and_date_llm_output.json", data)

        date_raw = (data.get("date") or "").strip()
        if date_raw and not re.fullmatch(r"\d{2}_\d{2}_\d{4}", date_raw):
            try:
                ff = _try_parse_date_fragment(date_raw)
                if ff:
                    _, _, y, m, d = ff[0]
                    date_raw = _normalize_date_mm_dd_yyyy(y, m, d)
            except Exception:
                pass
        if not date_raw:
            date_raw = _extract_service_date_heuristic(pdf_text) or ""

        date_folder = sanitize_title(date_raw) if date_raw else "unknown"
        new_state["order_folder"] = date_folder
        if _debug_enabled(new_state):
            base = _debug_dir_for(new_state)
            _write_text(base, "date_of_service.txt", date_raw or "unknown")

        # Normalize song entries + attach instrument (structure only)
        songs: Dict[int, Dict[str, Any]] = {}
        raw_songs = data.get("songs") if isinstance(data.get("songs"), (dict, list)) else {}
        if isinstance(raw_songs, list):
            normalized_songs = {i: v for i, v in enumerate(raw_songs)}
        elif isinstance(raw_songs, dict):
            normalized_songs = raw_songs
        else:
            normalized_songs = {}
        for k, v in normalized_songs.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            title = (v.get("title") or "").strip()
            artist = (v.get("artist") or "").strip()
            key = _normalize_extracted_key((v.get("key") or "").strip())
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


def process_songs_node(
    state: Dict[str, Any],
    progress_cb: Callable[[str, int, int], None] | None = None,
    event_cb: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Scrape all songs first (optionally in parallel), then batch process images.

    ``progress_cb`` receives stage progress, while ``event_cb`` can receive
    per-song status updates for UI streaming.
    """
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    debug_enabled = _debug_enabled(new_state)
    debug_base = _debug_dir_for(new_state) if debug_enabled else None
    if debug_enabled and debug_base is not None:
        _ensure_file_handler(new_state, debug_base)
        _logger.debug("process_songs_node: begin")

    if _cancel_requested(new_state):
        _mark_cancelled(new_state, "process_songs_node")
        new_state["final_pdfs"] = []
        _record_timing(new_state, "process_songs_node", _stop_timer(start))
        return new_state

    songs: Dict[int, Dict[str, Any]] = new_state.get("songs", {}) or {}
    song_timings: Dict[int, Dict[str, float]] = {
        idx: {"scrape": 0.0, "watermark": 0.0, "upscale": 0.0} for idx in songs
    }
    song_status: Dict[int, str] = {idx: "queued" for idx in songs}
    failed_songs: set[int] = set()
    scrape_starts: Dict[int, float] = {}
    final_pdfs: List[str | None] = []
    jobs: List[Dict[str, Any]] = []
    input_dir_override = new_state.get("input_dir", "data/samples")
    preview_root = str(new_state.get("preview_root") or "")
    if preview_root:
        try:
            os.makedirs(preview_root, exist_ok=True)
        except Exception:
            pass
    cancelled = False

    date_folder = _get_order_folder(new_state)
    try:
        _ensure_order_pdf_copied(new_state)
    except Exception as e:
        _record_error(new_state, "process_songs_node.ensure_order_pdf_copied_exception", e)

    def _report_progress(stage: str, done: int, total: int) -> None:
        if not progress_cb or total <= 0:
            return
        try:
            progress_cb(stage, done, total)
        except Exception:
            pass

    def _emit_event(payload: Dict[str, Any]) -> None:
        if not event_cb:
            return
        try:
            event_cb(payload)
        except Exception:
            pass

    def _emit_song_update(idx: int, status: str | None = None) -> None:
        if idx not in song_timings:
            return
        if status:
            song_status[idx] = status
        _emit_event(
            {
                "type": "song_update",
                "idx": idx,
                "status": song_status.get(idx, ""),
                "timings": dict(song_timings.get(idx, {})),
            }
        )

    for idx in sorted(songs.keys()):
        _emit_song_update(idx, "queued")

    parallel_flag = new_state.get("parallel_scrape")
    if isinstance(parallel_flag, bool):
        parallel_scrape = parallel_flag
    else:
        parallel_scrape = _env_truthy(os.getenv("ORDER_PARALLEL", ""))
    if parallel_scrape and songs:
        run_id = _get_run_id(new_state)
        run_ts = os.environ.get("RUN_TS") or run_id
        os.environ.setdefault("RUN_TS", run_ts)
        tasks: list[dict] = []
        per_song_names: dict[int, str] = {}
        total_scrapes = len(songs)
        _report_progress("scrape", 0, total_scrapes)
        for idx in sorted(songs.keys()):
            if _cancel_requested(new_state):
                cancelled = True
                _mark_cancelled(new_state, "process_songs_node")
                break
            song = songs[idx]
            safe_title = sanitize_title(song.get("title", "Unknown Title"))
            per_song_prefix = f"song_{str(idx + 1).zfill(2)}"
            per_song_name = f"{per_song_prefix}_{safe_title}"
            per_song_names[idx] = per_song_name
            scrape_starts[idx] = _start_timer()
            _emit_song_update(idx, "scraping")
            scrape_input = {
                "title": song.get("title", ""),
                "instrument": song.get("instrument", ""),
                "key": song.get("key", ""),
                "input_dir": input_dir_override,
                "artist": song.get("artist", ""),
                "top_n": int(new_state.get("top_n", 3) or 3),
            }
            if debug_enabled and debug_base is not None:
                _write_json(debug_base, f"{per_song_name}_scrape_input.json", scrape_input)
            log_dir = ""
            if debug_base is not None:
                log_dir = str((debug_base / "songs" / per_song_name))
            tasks.append({
                "idx": int(idx),
                "song": song,
                "input_dir": input_dir_override,
                "top_n": int(new_state.get("top_n", 3) or 3),
                "log_dir": log_dir,
                "preview_root": preview_root,
                "run_ts": run_ts,
            })
        if not cancelled:
            val = os.getenv("ORDER_MAX_PROCS", "")
            try:
                max_workers = int(val) if val.strip() else -1
            except Exception:
                max_workers = -1
            if max_workers <= 0:
                max_workers = len(tasks)
            else:
                max_workers = max(1, min(max_workers, len(tasks)))

            results: dict[int, list[dict[str, Any]]] = {}
            errors_all: list[dict[str, Any]] = []
            ctx = _mp.get_context("spawn")
            futures: dict[Any, int] = {}
            ex = _PPExecutor(max_workers=max_workers, mp_context=ctx)
            completed = 0
            try:
                futures = {ex.submit(_scrape_worker, t): t["idx"] for t in tasks}
                for fut in _as_completed(futures):
                    if _cancel_requested(new_state):
                        cancelled = True
                        _mark_cancelled(new_state, "process_songs_node")
                        for pending in futures:
                            pending.cancel()
                        _terminate_executor_processes(ex)
                        break
                    idx = futures[fut]
                    downloads: List[Dict[str, Any]] = []
                    try:
                        res = fut.result()
                    except Exception as e:
                        title = (songs.get(idx) or {}).get("title") or ""
                        msg = f"{title}: {e}" if title else str(e)
                        errors_all.append({"label": f"parallel_scrape_{idx + 1}.exception", "message": msg})
                        results[idx] = []
                    else:
                        if isinstance(res, dict):
                            downloads = list(res.get("downloads") or [])
                            results[idx] = downloads
                            for err in list(res.get("errors") or []):
                                title = (songs.get(idx) or {}).get("title") or ""
                                msg = f"{title}: {err}" if title else str(err)
                                errors_all.append({"label": f"parallel_scrape_{idx + 1}.error", "message": msg})
                        else:
                            results[idx] = []
                    elapsed = _stop_timer(scrape_starts.get(idx, _start_timer()))
                    song_timings[idx]["scrape"] = elapsed
                    if downloads:
                        _emit_song_update(idx, "downloaded")
                    else:
                        failed_songs.add(idx)
                        _emit_song_update(idx, "failed")
                    completed += 1
                    _report_progress("scrape", completed, total_scrapes)
            finally:
                if cancelled:
                    try:
                        ex.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                else:
                    ex.shutdown(wait=True)

            for err in errors_all:
                label = str(err.get("label") or "parallel_scrape_error")
                message = str(err.get("message") or err)
                _append_error(new_state, label, message)

            for idx in sorted(songs.keys()):
                song = songs[idx]
                per_song_name = per_song_names.get(idx) or f"song_{str(idx + 1).zfill(2)}"
                downloads = results.get(idx, [])
                if debug_enabled and debug_base is not None:
                    _write_text(debug_base, f"{per_song_name}_scrape_output.txt", str(downloads))
                _add_jobs_from_downloads(
                    new_state,
                    jobs,
                    final_pdfs,
                    idx=idx,
                    song=song,
                    per_song_name=per_song_name,
                    downloads=downloads,
                    debug_enabled=debug_enabled,
                    debug_base=debug_base,
                )
    else:
        total_scrapes = len(songs)
        completed = 0
        _report_progress("scrape", 0, total_scrapes)
        for idx in sorted(songs.keys()):
            if _cancel_requested(new_state):
                cancelled = True
                _mark_cancelled(new_state, "process_songs_node")
                break
            song = songs[idx]
            safe_title = sanitize_title(song.get("title", "Unknown Title"))
            per_song_prefix = f"song_{str(idx + 1).zfill(2)}"
            per_song_name = f"{per_song_prefix}_{safe_title}"
            scrape_start = _start_timer()
            scrape_starts[idx] = scrape_start
            _emit_song_update(idx, "scraping")

            try:
                if preview_root:
                    preview_dir = os.path.join(preview_root, f"song_{str(idx + 1).zfill(2)}")
                    try:
                        os.makedirs(preview_dir, exist_ok=True)
                        os.environ["WMRA_PREVIEW_DIR"] = preview_dir
                    except Exception:
                        os.environ.pop("WMRA_PREVIEW_DIR", None)
                else:
                    os.environ.pop("WMRA_PREVIEW_DIR", None)
                scrape_input = {
                    "title": song.get("title", ""),
                    "instrument": song.get("instrument", ""),
                    "key": song.get("key", ""),
                    "input_dir": input_dir_override,
                    "artist": song.get("artist", ""),
                    "top_n": int(new_state.get("top_n", 3) or 3),
                }
                if debug_enabled and debug_base is not None:
                    _write_json(debug_base, f"{per_song_name}_scrape_input.json", scrape_input)

                try:
                    from watermark_remover.agent.tools import init_pipeline_logging as _init_tools_log  # type: ignore
                    if debug_base is not None:
                        os.environ["WMRA_LOG_DIR"] = str((debug_base / "songs" / per_song_name))
                    _init_tools_log()
                except Exception:
                    pass

                try:
                    download_dir = scrape_music.invoke(scrape_input)
                except Exception as e:
                    download_dir = None
                    _record_error(new_state, f"{per_song_name}_scrape_exception", e)

                if debug_enabled and debug_base is not None:
                    _write_text(debug_base, f"{per_song_name}_scrape_output.txt", str(download_dir))

                downloads: List[Dict[str, Any]] = []
                if isinstance(download_dir, list):
                    for it in download_dir:
                        if not isinstance(it, dict):
                            continue
                        if it.get("image_dir") or it.get("existing_pdf"):
                            downloads.append(
                                {
                                    "image_dir": it.get("image_dir"),
                                    "existing_pdf": it.get("existing_pdf"),
                                    "meta": it.get("meta", {}),
                                }
                            )
                elif isinstance(download_dir, str):
                    downloads.append(
                        {
                            "image_dir": download_dir,
                            "meta": {
                                "title": SCRAPE_METADATA.get("title", song.get("title", "")),
                                "artist": SCRAPE_METADATA.get("artist", song.get("artist", "")),
                                "instrument": SCRAPE_METADATA.get("instrument", song.get("instrument", "")),
                                "key": SCRAPE_METADATA.get("key", song.get("key", "")),
                            },
                        }
                    )

                song_timings[idx]["scrape"] = _stop_timer(scrape_start)
                if downloads:
                    _emit_song_update(idx, "downloaded")
                else:
                    failed_songs.add(idx)
                    _emit_song_update(idx, "failed")
                _add_jobs_from_downloads(
                    new_state,
                    jobs,
                    final_pdfs,
                    idx=idx,
                    song=song,
                    per_song_name=per_song_name,
                    downloads=downloads,
                    debug_enabled=debug_enabled,
                    debug_base=debug_base,
                )

            except Exception as e:
                song_timings[idx]["scrape"] = _stop_timer(scrape_start)
                failed_songs.add(idx)
                _emit_song_update(idx, "failed")
                _record_error(new_state, f"{per_song_name}_outer_exception", e)
                final_pdfs.append(None)
            completed += 1
            _report_progress("scrape", completed, total_scrapes)

    if cancelled:
        new_state["final_pdfs"] = final_pdfs
        _record_timing(new_state, "process_songs_node", _stop_timer(start))
        return new_state

    if _cancel_requested(new_state):
        _mark_cancelled(new_state, "process_songs_node")
        new_state["final_pdfs"] = final_pdfs
        _record_timing(new_state, "process_songs_node", _stop_timer(start))
        return new_state

    jobs_by_song: Dict[int, int] = {}
    for job in jobs:
        try:
            song_idx = int(job.get("song_index", -1))
        except Exception:
            continue
        if song_idx < 0:
            continue
        jobs_by_song[song_idx] = jobs_by_song.get(song_idx, 0) + 1
    assemble_remaining: Dict[int, int] = dict(jobs_by_song)
    song_success: Dict[int, bool] = {idx: False for idx in jobs_by_song}

    # Batch watermark removal
    wm_jobs = [job for job in jobs if job.get("abs_download_dir")]
    if wm_jobs:
        abs_dirs = [job["abs_download_dir"] for job in wm_jobs if job.get("abs_download_dir")]
        wm_dir_to_song: Dict[str, int] = {}
        for job in wm_jobs:
            abs_dir = job.get("abs_download_dir")
            if not abs_dir:
                continue
            try:
                wm_dir_to_song[os.path.abspath(abs_dir)] = int(job.get("song_index", -1))
            except Exception:
                continue
        wm_started: set[int] = set()
        def wm_item_cb(
            phase: str,
            directory: str,
            _idx: int,
            _total: int,
            elapsed: float | None,
            _success: bool | None,
        ) -> None:
            song_idx = wm_dir_to_song.get(os.path.abspath(directory))
            if song_idx is None or song_idx < 0:
                return
            if song_idx in failed_songs:
                return
            if phase == "start":
                if song_idx not in wm_started:
                    wm_started.add(song_idx)
                    _emit_song_update(song_idx, "watermark")
                return
            if elapsed is not None:
                song_timings[song_idx]["watermark"] += float(elapsed)
            _emit_song_update(song_idx)
        total_wm = len(abs_dirs)
        _report_progress("watermark", 0, total_wm)
        wm_results, wm_errors = remove_watermark_batch(
            abs_dirs,
            progress_cb=lambda done, total: _report_progress("watermark", done, total),
            item_cb=wm_item_cb,
        )
        for job in wm_jobs:
            abs_dir = job.get("abs_download_dir")
            processed_dir = wm_results.get(abs_dir) if abs_dir else None
            job["processed_dir"] = processed_dir
            if processed_dir:
                job["abs_processed_dir"] = os.path.abspath(processed_dir)
            if debug_enabled and debug_base is not None:
                _write_text(
                    debug_base,
                    f"{job['per_song_name']}{job['variant_suffix']}_remove_watermark_output.txt",
                    str(processed_dir),
                )
            error = wm_errors.get(abs_dir) if abs_dir else None
            if error is not None:
                _record_error(new_state, f"{job['per_song_name']}{job['variant_suffix']}_remove_watermark_exception", error)
            elif abs_dir and processed_dir is None:
                _record_error(
                    new_state,
                    f"{job['per_song_name']}{job['variant_suffix']}_remove_watermark_exception",
                    RuntimeError("remove_watermark returned no output"),
                )

    if _cancel_requested(new_state):
        _mark_cancelled(new_state, "process_songs_node")
        new_state["final_pdfs"] = final_pdfs
        _record_timing(new_state, "process_songs_node", _stop_timer(start))
        return new_state

    # Batch upscaling
    up_jobs = [job for job in jobs if job.get("abs_processed_dir")]
    if up_jobs:
        processed_dirs = [job["abs_processed_dir"] for job in up_jobs if job.get("abs_processed_dir")]
        up_dir_to_song: Dict[str, int] = {}
        for job in up_jobs:
            abs_dir = job.get("abs_processed_dir")
            if not abs_dir:
                continue
            try:
                up_dir_to_song[os.path.abspath(abs_dir)] = int(job.get("song_index", -1))
            except Exception:
                continue
        up_started: set[int] = set()
        def up_item_cb(
            phase: str,
            directory: str,
            _idx: int,
            _total: int,
            elapsed: float | None,
            _success: bool | None,
        ) -> None:
            song_idx = up_dir_to_song.get(os.path.abspath(directory))
            if song_idx is None or song_idx < 0:
                return
            if song_idx in failed_songs:
                return
            if phase == "start":
                if song_idx not in up_started:
                    up_started.add(song_idx)
                    _emit_song_update(song_idx, "upscale")
                return
            if elapsed is not None:
                song_timings[song_idx]["upscale"] += float(elapsed)
            _emit_song_update(song_idx)
        total_up = len(processed_dirs)
        _report_progress("upscale", 0, total_up)
        up_results, up_errors = upscale_images_batch(
            processed_dirs,
            progress_cb=lambda done, total: _report_progress("upscale", done, total),
            item_cb=up_item_cb,
        )
        for job in up_jobs:
            abs_processed = job.get("abs_processed_dir")
            upscaled_dir = up_results.get(abs_processed) if abs_processed else None
            job["upscaled_dir"] = upscaled_dir
            if debug_enabled and debug_base is not None and abs_processed:
                _write_json(
                    debug_base,
                    f"{job['per_song_name']}{job['variant_suffix']}_upscale_input.json",
                    {"input_dir": job["processed_dir"]},
                )
                _write_text(
                    debug_base,
                    f"{job['per_song_name']}{job['variant_suffix']}_upscale_output.txt",
                    str(upscaled_dir),
                )
            error = up_errors.get(abs_processed) if abs_processed else None
            if error is not None:
                _record_error(new_state, f"{job['per_song_name']}{job['variant_suffix']}_upscale_exception", error)
            elif abs_processed and upscaled_dir is None:
                _record_error(
                    new_state,
                    f"{job['per_song_name']}{job['variant_suffix']}_upscale_exception",
                    RuntimeError("upscale_images returned no output"),
                )

    if _cancel_requested(new_state):
        _mark_cancelled(new_state, "process_songs_node")
        new_state["final_pdfs"] = final_pdfs
        _record_timing(new_state, "process_songs_node", _stop_timer(start))
        return new_state

    # Assemble sequentially to honour user-facing order
    for job in jobs:
        if _cancel_requested(new_state):
            _mark_cancelled(new_state, "process_songs_node")
            break
        result_index = job["result_index"]
        pdf_path: Optional[str] = None
        upscaled_dir = job.get("upscaled_dir")
        existing_pdf = job.get("existing_pdf")
        if upscaled_dir:
            as_input = {
                "image_dir": upscaled_dir,
                "output_pdf": job["filename"],
                "meta": job["assembly_meta"],
            }
            if debug_enabled and debug_base is not None:
                _write_json(
                    debug_base,
                    f"{job['per_song_name']}{job['variant_suffix']}_assemble_input.json",
                    as_input,
                )
            try:
                pdf_path = assemble_pdf.invoke(as_input)
            except Exception as e:
                _record_error(new_state, f"{job['per_song_name']}{job['variant_suffix']}_assemble_exception", e)
                pdf_path = None
            else:
                if debug_enabled and debug_base is not None:
                    _write_text(
                        debug_base,
                        f"{job['per_song_name']}{job['variant_suffix']}_assemble_output.txt",
                        str(pdf_path),
                    )
            if pdf_path:
                try:
                    import shutil

                    orders_root = Path(os.getcwd()) / "output" / "orders" / date_folder
                    orders_root.mkdir(parents=True, exist_ok=True)
                    target_path = orders_root / job["filename"]
                    shutil.copyfile(pdf_path, str(target_path))
                    pdf_path = str(target_path)
                    if debug_enabled and debug_base is not None:
                        _write_text(
                            debug_base,
                            f"{job['per_song_name']}{job['variant_suffix']}_final_target.txt",
                            str(target_path),
                        )
                except Exception as e:
                    _record_error(new_state, f"{job['per_song_name']}{job['variant_suffix']}_copy_exception", e)
        elif existing_pdf:
            pdf_path = str(existing_pdf)

        final_pdfs[result_index] = pdf_path if isinstance(pdf_path, str) else None
        try:
            song_idx = int(job.get("song_index", -1))
        except Exception:
            song_idx = -1
        if song_idx >= 0:
            if pdf_path:
                song_success[song_idx] = True
            if song_idx in assemble_remaining:
                assemble_remaining[song_idx] -= 1
                if assemble_remaining[song_idx] <= 0 and song_idx not in failed_songs:
                    final_status = "done" if song_success.get(song_idx) else "failed"
                    if final_status == "failed":
                        failed_songs.add(song_idx)
                    _emit_song_update(song_idx, final_status)

    # Cleanup temporary directories once after all processing completes
    try:
        import shutil

        removed: List[str] = []
        for tmp_dir in _iter_temp_dirs():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                removed.append(tmp_dir)
            except Exception:
                pass
        TEMP_DIRS.clear()
        if debug_enabled and debug_base is not None:
            _write_text(
                debug_base,
                "temp_cleanup.txt",
                f"Removed {len(removed)} temporary directories.",
            )
    except Exception as e:
        _record_error(new_state, "process_songs_node.temp_cleanup_exception", e)

    new_state["final_pdfs"] = final_pdfs

    if debug_enabled and debug_base is not None:
        _write_json(
            debug_base,
            "final_state.json",
            {k: v for k, v in new_state.items() if k not in ("songs", "_cancel_event")},
        )

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
        # Ensure 00_<Month>_<DD>_<YYYY>_Order_Of_Worship.pdf exists in output/orders/<date>
        try:
            _ensure_order_pdf_copied(new_state)
        except Exception as e:
            _record_error(new_state, "init_loop_node.ensure_order_pdf_copied_exception", e)

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
    # Ensure tools' pipeline logs are set up for sequential mode
    try:
        from watermark_remover.agent.tools import init_pipeline_logging as _init_tools_log  # type: ignore
        # Set per-song WMRA_LOG_DIR for sequential runs to unify artifacts
        idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)
        base = _debug_dir_for(new_state)
        import os as _os
        _os.environ["WMRA_LOG_DIR"] = str((base / "songs" / per_song_name))
        _init_tools_log()
    except Exception:
        pass

    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)
    input_dir_override = new_state.get("input_dir", "data/samples")

    scrape_input = {
        "title": song.get("title", ""),
        "instrument": song.get("instrument", ""),
        "key": song.get("key", ""),
        "input_dir": input_dir_override,
        "artist": song.get("artist", ""),
        "top_n": int(new_state.get("top_n", 3) or 3),
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

    # Normalize into a list of downloads for downstream nodes
    downloads: List[Dict[str, Any]] = []
    if isinstance(download_dir, list):
        for it in download_dir:
            if not isinstance(it, dict):
                continue
            if it.get("image_dir"):
                downloads.append({"image_dir": it.get("image_dir"), "meta": it.get("meta", {})})
            elif it.get("existing_pdf"):
                downloads.append({"existing_pdf": it.get("existing_pdf"), "meta": it.get("meta", {})})
    elif isinstance(download_dir, str):
        downloads.append({
            "image_dir": download_dir,
            "meta": {
                "title": SCRAPE_METADATA.get("title", song.get("title", "")),
                "artist": SCRAPE_METADATA.get("artist", song.get("artist", "")),
                "instrument": SCRAPE_METADATA.get("instrument", song.get("instrument", "")),
                "key": SCRAPE_METADATA.get("key", song.get("key", "")),
            },
        })
    new_state["downloads"] = downloads
    _record_timing(new_state, "scraper_node", _stop_timer(start))
    return new_state


def watermark_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: watermark removal for all matches of the current song.

    Consumes a list in state['downloads'] with items {image_dir, meta} and adds
    'processed_dir' per item. Keeps backward compatibility if only a single
    'download_dir' is present in the state.
    """
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)
    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)

    downloads: List[Dict[str, Any]] = new_state.get("downloads", []) or []
    # Back-compat: if only a single download_dir, convert it
    if not downloads and new_state.get("download_dir"):
        downloads = [{"image_dir": new_state.get("download_dir"), "meta": {}}]
    updated: List[Dict[str, Any]] = []
    for i, dl in enumerate(downloads, 1):
        img_dir = dl.get("image_dir")
        if not img_dir:
            updated.append(dl)
            continue
        try:
            rm_input = {"input_dir": img_dir}
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_match{str(i).zfill(2)}_remove_watermark_input.json", rm_input)
            processed_dir = remove_watermark.invoke(rm_input)
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_match{str(i).zfill(2)}_remove_watermark_output.txt", str(processed_dir))
            dl = dict(dl)
            dl["processed_dir"] = processed_dir
        except Exception as e:
            _record_error(new_state, f"{per_song_name}_match{str(i).zfill(2)}_remove_watermark_exception", e)
        updated.append(dl)

    new_state["downloads"] = updated
    _record_timing(new_state, "watermark_node", _stop_timer(start))
    return new_state


def upscaler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: upscaling for all matches; consumes processed_dir and adds upscaled_dir per item."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)
    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)

    downloads: List[Dict[str, Any]] = new_state.get("downloads", []) or []
    updated: List[Dict[str, Any]] = []
    for i, dl in enumerate(downloads, 1):
        proc_dir = dl.get("processed_dir")
        if not proc_dir:
            updated.append(dl)
            continue
        try:
            up_input = {"input_dir": proc_dir}
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_json(base, f"{per_song_name}_match{str(i).zfill(2)}_upscale_input.json", up_input)
            upscaled_dir = upscale_images.invoke(up_input)
            if _debug_enabled(new_state):
                base = _debug_dir_for(new_state)
                _write_text(base, f"{per_song_name}_match{str(i).zfill(2)}_upscale_output.txt", str(upscaled_dir))
            dl = dict(dl)
            dl["upscaled_dir"] = upscaled_dir
        except Exception as e:
            _record_error(new_state, f"{per_song_name}_match{str(i).zfill(2)}_upscale_exception", e)
        updated.append(dl)

    new_state["downloads"] = updated
    _record_timing(new_state, "upscaler_node", _stop_timer(start))
    return new_state


def assembler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: assemble PDFs for all matches of the current song; then advance index and cleanup temps."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)

    idx, song, per_song_prefix, per_song_name = _per_song_labels(new_state)
    downloads: List[Dict[str, Any]] = new_state.get("downloads", []) or []
    date_folder = new_state.get("_date_folder") or _get_order_folder(new_state)
    final_pdfs: List[str | None] = list(new_state.get("final_pdfs", []))
    for match_idx, dl in enumerate(downloads, 1):
        upscaled_dir = dl.get("upscaled_dir")
        existing_pdf = dl.get("existing_pdf")
        pdf_path: str | None = None
        if upscaled_dir:
            try:
                # Use per-match meta for naming (avoid global state)
                m = dl.get("meta") or {}
                actual_key = m.get("key") or song.get("key", "")
                actual_instrument = m.get("instrument") or song.get("instrument", "")
                actual_title_for_name = m.get("title") or song.get("title", "Unknown Title")
                safe_instrument = sanitize_title(actual_instrument or "Unknown Instrument")
                safe_key = sanitize_title(actual_key or "Unknown Key")
                safe_title_for_name = sanitize_title(actual_title_for_name)
                idx_str = str(idx + 1).zfill(2)
                rank_str = str(match_idx).zfill(2)
                filename = f"{idx_str}_{rank_str}_{safe_title_for_name}_{safe_instrument}_{safe_key}.pdf"

                as_input = {"image_dir": upscaled_dir, "output_pdf": filename, "meta": {
                    "title": actual_title_for_name,
                    "artist": m.get("artist", ""),
                    "instrument": actual_instrument,
                    "key": actual_key,
                }}
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_json(base, f"{per_song_name}_match{rank_str}_assemble_input.json", as_input)
                pdf_path = assemble_pdf.invoke(as_input)
                if _debug_enabled(new_state):
                    base = _debug_dir_for(new_state)
                    _write_text(base, f"{per_song_name}_match{rank_str}_assemble_output.txt", str(pdf_path))

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
                            _write_text(base, f"{per_song_name}_match{rank_str}_final_target.txt", str(target_path))
                    except Exception as e:
                        _record_error(new_state, f"{per_song_name}_match{rank_str}_copy_exception", e)
            except Exception as e:
                _record_error(new_state, f"{per_song_name}_match{str(match_idx).zfill(2)}_assemble_outer_exception", e)
        final_pdfs.append(pdf_path if isinstance(pdf_path, str) else None)
    new_state["final_pdfs"] = final_pdfs

    # Cleanup TEMP_DIRS between songs
    try:
        import shutil as _shutil
        for tmp_dir in _iter_temp_dirs():
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


def process_songs_parallel_node(
    state: Dict[str, Any],
    progress_cb: Callable[[str, int, int], None] | None = None,
    event_cb: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Scrape in parallel, then batch watermark removal/upscaling in order."""
    _ensure_logger_configured(logging.DEBUG if _debug_enabled(state) else logging.INFO)
    start = _start_timer()
    new_state: Dict[str, Any] = dict(state)
    new_state["parallel_scrape"] = True
    result = process_songs_node(new_state, progress_cb=progress_cb, event_cb=event_cb)
    _record_timing(result, "process_songs_parallel_node", _stop_timer(start))
    return result


def compile_graph() -> Any:
    """Construct and compile the order-of-worship graph.

    By default, scraping runs in parallel using separate processes for Selenium
    (ORDER_PARALLEL truthy), then watermark removal/upscaling run in batch order.
    Set ORDER_PARALLEL=0 to keep scraping sequential.
    """
    graph = StateGraph(dict)
    # Shared prefix
    graph.add_node("parser", parser_node)
    graph.add_node("extractor", extract_songs_node)
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "extractor")

    use_parallel = _env_truthy(os.getenv("ORDER_PARALLEL", "1"))
    if use_parallel:
        graph.add_node("process_parallel", process_songs_parallel_node)
        graph.add_edge("extractor", "process_parallel")
        graph.add_edge("process_parallel", END)
    else:
        graph.add_node("process_batch", process_songs_node)
        graph.add_edge("extractor", "process_batch")
        graph.add_edge("process_batch", END)

    return graph.compile()

# Expose compiled graph
_DEFAULT_RL = int(os.getenv("ORDER_RECURSION_LIMIT", "100"))
graph = compile_graph().with_config({"recursion_limit": _DEFAULT_RL})
