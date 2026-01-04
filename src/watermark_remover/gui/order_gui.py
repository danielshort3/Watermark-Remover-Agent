"""Gradio GUI for order-of-worship workflows."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import gradio as gr  # type: ignore
except Exception as exc:  # pragma: no cover - import-time dependency check
    raise ImportError(
        "Gradio is required for the GUI. Install it with `pip install gradio`."
    ) from exc

from config.settings import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL
from watermark_remover.agent import order_of_worship_graph as order_graph
from watermark_remover.agent import single_song_graph as single_song_graph
from watermark_remover.agent.graph_ollama import _ping_ollama as ping_ollama
from watermark_remover.agent.tools import (
    assemble_pdf,
    remove_watermark,
    sanitize_title,
    scrape_music,
    upscale_images,
)

LOGGER = logging.getLogger(__name__)

TABLE_HEADERS = ["Include", "Order", "Title", "Artist", "Key", "Instrument"]
TABLE_TYPES = ["bool", "number", "str", "str", "str", "str"]
DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
MODEL_PLACEHOLDER = "(connect to Ollama to load models)"
_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENT: Any | None = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _new_run_ts() -> str:
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TS"] = run_ts
    return run_ts


def _normalize_model_choice(value: str) -> str:
    if not value:
        return ""
    if value.strip() == MODEL_PLACEHOLDER:
        return ""
    return value.strip()


def _new_cancel_event(use_multiproc: bool) -> Any:
    event = threading.Event()
    _set_cancel_event(event)
    return event


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _set_cancel_event(event: Any | None) -> None:
    global _CANCEL_EVENT
    with _CANCEL_LOCK:
        _CANCEL_EVENT = event


def _get_cancel_event() -> Any | None:
    with _CANCEL_LOCK:
        return _CANCEL_EVENT


def _cancel_requested() -> bool:
    event = _get_cancel_event()
    if event is None or not hasattr(event, "is_set"):
        return False
    try:
        return bool(event.is_set())
    except Exception:
        return False


def _apply_capture_env(
    save_screens: bool,
    save_html: bool,
    errors_only: bool,
) -> None:
    os.environ["WMRA_SAVE_SCREENSHOTS"] = "1" if save_screens else "0"
    os.environ["WMRA_SAVE_HTML"] = "1" if save_html else "0"
    os.environ["WMRA_SAVE_ON_ERROR_ONLY"] = "1" if errors_only else "0"


def _resolve_ollama_env(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
) -> Tuple[str, str, str, str]:
    host = _safe_str(ollama_host) or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    url = _safe_str(ollama_url) or os.environ.get("OLLAMA_URL", "")
    if not url:
        url = f"http://{host}" if host else DEFAULT_OLLAMA_URL
    model = _normalize_model_choice(_safe_str(ollama_model)) or os.environ.get(
        "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL
    )
    models_path = _safe_str(ollama_models_path) or os.environ.get("OLLAMA_MODELS", "")
    if not models_path:
        models_path = _detect_ollama_models_path()
    return url, model, host, models_path


def _apply_ollama_env(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
) -> Tuple[str, str, str, str]:
    url, model, host, models_path = _resolve_ollama_env(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
    )
    os.environ["OLLAMA_URL"] = url
    if model:
        os.environ["OLLAMA_MODEL"] = model
    if host:
        os.environ["OLLAMA_HOST"] = host
    if models_path:
        os.environ["OLLAMA_MODELS"] = models_path
    return url, model, host, models_path


def _looks_like_ollama_models_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        return (path / "blobs").is_dir() and (path / "manifests").is_dir()
    except Exception:
        return False


def _detect_ollama_models_path() -> str:
    base = Path("/mnt/c/Users")
    if not base.is_dir():
        return ""
    try:
        users = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return ""

    for user_dir in sorted(users, key=lambda p: p.name.lower()):
        candidate = user_dir / ".ollama" / "models"
        if _looks_like_ollama_models_dir(candidate):
            return str(candidate)

    for user_dir in sorted(users, key=lambda p: p.name.lower()):
        candidate = user_dir / "AppData" / "Local" / "Ollama" / "models"
        if _looks_like_ollama_models_dir(candidate):
            return str(candidate)

    return ""


def _discover_models_from_disk(models_path: str) -> List[str]:
    if not models_path:
        return []
    base = Path(models_path) / "manifests" / "registry.ollama.ai" / "library"
    if not base.is_dir():
        return []
    models: List[str] = []
    try:
        for name_dir in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if not name_dir.is_dir():
                continue
            for tag in sorted(name_dir.iterdir(), key=lambda p: p.name.lower()):
                if not tag.is_dir():
                    continue
                models.append(f"{name_dir.name}:{tag.name}")
    except Exception:
        return models
    return models


def _choose_default_model(models: List[str], fallback: str) -> str:
    if fallback and fallback in models:
        return fallback
    if DEFAULT_OLLAMA_MODEL in models:
        return DEFAULT_OLLAMA_MODEL
    return models[0] if models else fallback or DEFAULT_OLLAMA_MODEL


def _refresh_models(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    diag: Dict[str, Any] | None = None,
) -> Tuple[List[str], str | None]:
    url, model, host, models_path = _apply_ollama_env(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
    )
    if diag is None:
        diag = ping_ollama(url, model)
    if not diag.get("ok", False):
        return [MODEL_PLACEHOLDER], MODEL_PLACEHOLDER
    models = (
        [m for m in diag.get("models") if isinstance(m, str)]
        if isinstance(diag.get("models"), list)
        else []
    )
    if not models:
        return [MODEL_PLACEHOLDER], MODEL_PLACEHOLDER
    chosen = _choose_default_model(models, model)
    return models, chosen


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _llm_trace_path() -> Path:
    base = os.environ.get("WMRA_LOG_DIR")
    if not base:
        run_ts = os.environ.get("RUN_TS")
        if not run_ts:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            os.environ.setdefault("RUN_TS", run_ts)
        base = os.path.join(os.getcwd(), "output", "logs", run_ts)
    return Path(base) / "llm_trace.jsonl"


def _read_llm_trace_tail(max_lines: int = 30) -> str:
    path = _llm_trace_path()
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n".join(tail)


def _model_manifest_paths(models_path: str, model: str) -> List[Path]:
    if not models_path or not model:
        return []
    base = Path(models_path) / "manifests" / "registry.ollama.ai" / "library"
    if ":" in model:
        name, tag = model.split(":", 1)
        return [base / name / tag]
    return [base / model / "latest", base / model]


def _summarize_ollama_list(env: Dict[str, str]) -> str:
    if not shutil.which("ollama"):
        return "ollama list: binary not found"
    try:
        result = subprocess.run(
            ["ollama", "list"],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return f"ollama list: failed ({exc})"
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        return f"ollama list: exit {result.returncode} {err or out}".strip()
    if not out:
        return "ollama list: no output"
    lines = out.splitlines()
    if len(lines) > 12:
        lines = lines[:12] + ["... (truncated)"]
    return "ollama list:\n" + "\n".join(lines)


def _build_ollama_debug(
    url: str,
    model: str,
    host: str,
    models_path: str,
    diag: Dict[str, Any],
    proc_state: Dict[str, Any] | None,
    include_ollama_list: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"OLLAMA_URL={url}")
    lines.append(f"OLLAMA_HOST={host or '(unset)'}")
    lines.append(f"OLLAMA_MODEL={model}")
    lines.append(f"OLLAMA_MODELS={models_path or '(unset)'}")
    lines.append(f"llm_trace={_llm_trace_path()}")

    models_dir = Path(models_path) if models_path else None
    if models_dir and models_dir.exists():
        lines.append(f"models_path exists: yes ({'dir' if models_dir.is_dir() else 'file'})")
    else:
        lines.append("models_path exists: no")

    manifest_paths = _model_manifest_paths(models_path, model)
    for mp in manifest_paths:
        lines.append(f"manifest check: {mp} -> {'yes' if mp.exists() else 'no'}")

    ollama_bin = shutil.which("ollama") or ""
    lines.append(f"ollama binary: {ollama_bin or 'not found'}")

    if proc_state:
        pid = _safe_int(proc_state.get("pid"))
        lines.append(f"ollama pid: {pid or 'n/a'} (alive={_process_alive(pid)})")
        log_path = _safe_str(proc_state.get("log_path"))
        if log_path:
            lines.append(f"ollama log: {log_path}")

    if diag:
        lines.append(f"ping ok: {bool(diag.get('ok'))}")
        lines.append(f"has model: {bool(diag.get('has_model'))}")
        models = diag.get("models") if isinstance(diag.get("models"), list) else []
        if models:
            lines.append(f"models seen: {_format_model_list(models)}")

    if include_ollama_list:
        env = os.environ.copy()
        if host:
            env["OLLAMA_HOST"] = host
        if models_path:
            env["OLLAMA_MODELS"] = models_path
        lines.append(_summarize_ollama_list(env))

    return "\n".join(lines)


def _format_model_list(models: List[Any] | None) -> str:
    if not models:
        return "(none)"
    cleaned = [str(m).strip() for m in models if m]
    if not cleaned:
        return "(none)"
    max_list = 12
    trimmed = cleaned[:max_list]
    extra = len(cleaned) - max_list
    if extra > 0:
        trimmed.append(f"... (+{extra} more)")
    return ", ".join(trimmed)


def _format_ollama_status(diag: Dict[str, Any]) -> str:
    base_url = _safe_str(diag.get("base_url"))
    model = _safe_str(diag.get("model"))
    if diag.get("ok", False):
        if diag.get("has_model", False):
            return f"Ollama OK at {base_url}. Model '{model}' found."
        models = diag.get("models") if isinstance(diag.get("models"), list) else []
        return (
            f"Ollama OK at {base_url} but model '{model}' not found. "
            f"Available models: {_format_model_list(models)}"
        )
    error = _safe_str(diag.get("error")) or "Unknown error"
    return f"Ollama unreachable at {base_url}: {error}"


def _collect_preview_images(preview_root: str) -> List[Tuple[str, str]]:
    if not preview_root or not os.path.isdir(preview_root):
        return []
    items: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(preview_root)):
        worker_dir = os.path.join(preview_root, entry)
        if not os.path.isdir(worker_dir):
            continue
        screenshots_dir = os.path.join(worker_dir, "screenshots")
        if not os.path.isdir(screenshots_dir):
            continue
        latest_path = ""
        latest_mtime = -1.0
        try:
            for fname in os.listdir(screenshots_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                path = os.path.join(screenshots_dir, fname)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    continue
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path
        except Exception:
            continue
        if latest_path:
            items.append((latest_path, entry))
    return items


def cancel_processing() -> Tuple[str, None]:
    event = _get_cancel_event()
    if event is None:
        return "No active run to cancel.", None
    try:
        event.set()
    except Exception:
        return "Cancel requested, but the signal could not be sent.", None
    return "Cancel requested. Waiting for the current step to finish.", None


def cancel_order_processing() -> Tuple[str, None, List[Tuple[str, str]]]:
    message, _ = cancel_processing()
    return message, None, []




def check_ollama(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    proc_state: Dict[str, Any] | None,
) -> Tuple[str, str, gr.Dropdown]:
    url, model, host, models_path = _apply_ollama_env(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
    )
    diag = ping_ollama(url, model)
    debug_text = ""
    if ollama_debug:
        debug_text = _build_ollama_debug(
            url,
            model,
            host,
            models_path,
            diag,
            proc_state,
            include_ollama_list=True,
        )
    models, chosen = _refresh_models(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
        diag=diag,
    )
    return _format_ollama_status(diag), debug_text, gr.update(
        choices=models,
        value=chosen if chosen else None,
    )


def start_ollama_server(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    proc_state: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], str, str]:
    url, model, host, models_path = _apply_ollama_env(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
    )
    diag = ping_ollama(url, model)
    if diag.get("ok", False):
        debug_text = ""
        if ollama_debug:
            debug_text = _build_ollama_debug(
                url,
                model,
                host,
                models_path,
                diag,
                proc_state,
                include_ollama_list=True,
            )
        return proc_state or {}, _format_ollama_status(diag), debug_text

    if proc_state and _process_alive(_safe_int(proc_state.get("pid"))):
        pid = _safe_int(proc_state.get("pid"))
        status = f"Ollama already running (pid {pid})."
        debug_text = ""
        if ollama_debug:
            debug_text = _build_ollama_debug(
                url,
                model,
                host,
                models_path,
                diag,
                proc_state,
                include_ollama_list=True,
            )
        return proc_state, status, debug_text

    if not shutil.which("ollama"):
        status = "ollama binary not found in PATH for WSL."
        debug_text = ""
        if ollama_debug:
            debug_text = _build_ollama_debug(
                url,
                model,
                host,
                models_path,
                diag,
                proc_state,
                include_ollama_list=False,
            )
        return proc_state or {}, status, debug_text

    log_dir = Path(os.getcwd()) / "output" / "ollama"
    _ensure_dir(log_dir)
    log_path = log_dir / "ollama_server.log"
    missing_models = bool(models_path) and not os.path.isdir(models_path)

    env = os.environ.copy()
    if host:
        env["OLLAMA_HOST"] = host
    if models_path:
        env["OLLAMA_MODELS"] = models_path

    with open(log_path, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(0.6)
    diag = ping_ollama(url, model)
    status = _format_ollama_status(diag)
    if not diag.get("ok", False):
        status = f"Started ollama (pid {proc.pid}). {status}"
    else:
        status = f"Started ollama (pid {proc.pid}). {status}"
    if models_path:
        status += f" Models path: {models_path}"
    if missing_models:
        status += " WARNING: OLLAMA_MODELS path not found."
    status += f" Logs: {log_path}"

    new_state = {
        "pid": proc.pid,
        "host": host,
        "models_path": models_path,
        "log_path": str(log_path),
    }
    debug_text = ""
    if ollama_debug:
        debug_text = _build_ollama_debug(
            url,
            model,
            host,
            models_path,
            diag,
            new_state,
            include_ollama_list=True,
        )
    return new_state, status, debug_text


def stop_ollama_server(
    proc_state: Dict[str, Any] | None,
    ollama_debug: bool,
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
) -> Tuple[Dict[str, Any], str, str]:
    if not proc_state:
        return {}, "No Ollama process started by the GUI.", ""
    pid = _safe_int(proc_state.get("pid"))
    if not _process_alive(pid):
        return {}, "Ollama process is not running.", ""
    try:
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                os.kill(pid, signal.SIGTERM)
        status = f"Stopped ollama (pid {pid})."
        debug_text = ""
        if ollama_debug:
            url, model, host, models_path = _apply_ollama_env(
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
            )
            diag = ping_ollama(url, model)
            debug_text = _build_ollama_debug(
                url,
                model,
                host,
                models_path,
                diag,
                None,
                include_ollama_list=False,
            )
        return {}, status, debug_text
    except Exception as exc:
        return proc_state, f"Failed to stop ollama: {exc}", ""


def force_restart_ollama(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    proc_state: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], str, str]:
    if shutil.which("ollama"):
        try:
            subprocess.run(
                ["pkill", "-f", "ollama serve"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    return start_ollama_server(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
        ollama_debug,
        {},
    )


def auto_start_ollama(
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    proc_state: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], str, str, gr.Dropdown]:
    proc_state, start_status, start_debug = start_ollama_server(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
        ollama_debug,
        proc_state,
    )
    status, debug_text, model_update = check_ollama(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
        ollama_debug,
        proc_state,
    )
    if "Ollama OK" not in status and start_status:
        status = start_status
    if not debug_text and start_debug:
        debug_text = start_debug
    return proc_state, status, debug_text, model_update


def _upload_root() -> Path:
    return Path(os.getcwd()) / "input" / "uploads"


def _store_upload(src_path: str) -> str:
    if not src_path:
        raise ValueError("No upload path provided.")
    root = _upload_root()
    _ensure_dir(root)
    src = Path(src_path)
    suffix = src.suffix if src.suffix else ".pdf"
    stem = sanitize_title(src.stem) or "order"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = root / f"{stem}_{timestamp}{suffix}"
    shutil.copyfile(src_path, target)
    LOGGER.info("Saved upload to %s", target)
    return str(target)


def _store_image_uploads(file_paths: List[str], run_ts: str) -> str:
    if not file_paths:
        raise ValueError("No image uploads provided.")
    root = Path(os.getcwd()) / "output" / "manual" / run_ts / "1_original"
    _ensure_dir(root)
    copied = 0
    for idx, raw in enumerate(file_paths, start=1):
        if not raw:
            continue
        src = Path(raw)
        suffix = src.suffix if src.suffix else ".png"
        if suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        stem = sanitize_title(src.stem) or "page"
        target = root / f"{idx:03d}_{stem}{suffix}"
        shutil.copyfile(raw, target)
        copied += 1
    if copied == 0:
        raise ValueError("No valid image files found in upload.")
    LOGGER.info("Saved %d uploaded image(s) to %s", copied, root)
    return str(root)


def _build_song_rows(
    songs: Dict[int, Dict[str, Any]],
    default_instrument: str,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for idx in sorted(songs.keys()):
        song = songs[idx] or {}
        rows.append(
            [
                True,
                idx + 1,
                _safe_str(song.get("title")),
                _safe_str(song.get("artist")),
                _safe_str(song.get("key")),
                _safe_str(song.get("instrument")) or default_instrument,
            ]
        )
    return rows


def _coerce_rows(rows: Any) -> List[List[Any]]:
    if rows is None:
        return []
    if isinstance(rows, (str, bytes)):
        return []
    if hasattr(rows, "values") and hasattr(rows, "columns"):
        try:
            return rows.values.tolist()
        except Exception:
            pass
    if hasattr(rows, "tolist"):
        try:
            data = rows.tolist()
            if isinstance(data, list):
                return data
        except Exception:
            pass
    if isinstance(rows, list):
        return rows
    if isinstance(rows, tuple):
        return list(rows)
    try:
        return list(rows)
    except Exception:
        return []


def _coerce_file_list(files: Any) -> List[str]:
    if not files:
        return []
    if isinstance(files, str):
        return [files]
    if isinstance(files, list):
        return [f for f in files if f]
    if isinstance(files, tuple):
        return [f for f in files if f]
    try:
        return [f for f in list(files) if f]
    except Exception:
        return []


def _parse_song_rows(
    rows: Iterable[Iterable[Any]] | None,
    default_instrument: str,
) -> Dict[int, Dict[str, str]]:
    rows_list = _coerce_rows(rows)
    if not rows_list:
        return {}
    normalized: List[Tuple[int | None, int, Dict[str, str]]] = []
    for pos, raw in enumerate(rows_list):
        row = list(raw) if isinstance(raw, (list, tuple)) else []
        if len(row) < 6:
            continue
        include = _safe_bool(row[0])
        if not include:
            continue
        order_val = _safe_int(row[1])
        title = _safe_str(row[2])
        if not title:
            continue
        artist = _safe_str(row[3])
        key = _safe_str(row[4])
        instrument = _safe_str(row[5]) or default_instrument
        normalized.append(
            (
                order_val,
                pos,
                {
                    "title": title,
                    "artist": artist,
                    "key": key,
                    "instrument": instrument,
                },
            )
        )
    normalized.sort(key=lambda item: (item[0] is None, item[0] or item[1], item[1]))
    songs: Dict[int, Dict[str, str]] = {}
    for idx, (_, _, payload) in enumerate(normalized):
        songs[idx] = payload
    return songs


def _order_dir(order_folder: str | None) -> Path:
    folder = sanitize_title(order_folder or "") or "unknown_date"
    return Path(os.getcwd()) / "output" / "orders" / folder


def _find_order_pdf(order_dir: Path) -> Path | None:
    candidates = sorted(
        order_dir.glob("00_*_Order_Of_Worship.pdf"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _build_zip(
    order_folder: str | None,
    run_id: str,
    final_pdfs: List[str | None],
) -> str | None:
    order_dir = _order_dir(order_folder)
    if not order_dir.exists():
        return None

    files: List[Path] = []
    order_pdf = _find_order_pdf(order_dir)
    if order_pdf:
        files.append(order_pdf)

    for pdf in final_pdfs:
        if not pdf:
            continue
        path = Path(pdf)
        if not path.exists():
            candidate = order_dir / path.name
            if candidate.exists():
                path = candidate
            else:
                continue
        files.append(path)

    if not files:
        return None

    seen: set[str] = set()
    zip_name = f"{sanitize_title(order_folder or 'order') or 'order'}_{run_id}.zip"
    zip_path = order_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            archive.write(path, arcname=path.name)
    return str(zip_path)


def _summarize_errors(state: Dict[str, Any], prior_count: int) -> Tuple[int, List[str]]:
    errors = state.get("errors", []) or []
    new_errors = errors[prior_count:]
    messages: List[str] = []
    for err in new_errors:
        label = _safe_str(err.get("label") if isinstance(err, dict) else "error")
        msg = _safe_str(err.get("message") if isinstance(err, dict) else err)
        if label or msg:
            messages.append(f"ERROR {label}: {msg}".strip())
    return len(errors), messages


def analyze_order(
    pdf_file: str | None,
    instruction: str,
    default_instrument: str,
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    debug: bool,
) -> Tuple[List[List[Any]], Dict[str, Any], str, str, None, str, str]:
    """Extract songs from an order-of-worship PDF for review."""
    if not pdf_file:
        return [], {}, "Upload a PDF to begin.", "", None, "", ""

    try:
        stored_path = _store_upload(pdf_file)
    except Exception as exc:
        LOGGER.exception("Failed to store upload")
        return [], {}, f"Failed to store upload: {exc}", "", None, "", ""

    url, model, host, models_path = _apply_ollama_env(
        ollama_url,
        ollama_model,
        ollama_host,
        ollama_models_path,
    )
    if ollama_debug:
        os.environ["WMRA_LLM_TRACE"] = "1"
        os.environ["WMRA_LLM_TRACE_VERBOSE"] = "1"
    else:
        os.environ.pop("WMRA_LLM_TRACE", None)
        os.environ.pop("WMRA_LLM_TRACE_VERBOSE", None)
    diag = ping_ollama(url, model)
    health_status = _format_ollama_status(diag)
    debug_text = ""
    if ollama_debug:
        debug_text = _build_ollama_debug(
            url,
            model,
            host,
            models_path,
            diag,
            None,
            include_ollama_list=True,
        )
    if not diag.get("ok", False) or not diag.get("has_model", False):
        if debug_text:
            tail = _read_llm_trace_tail()
            if tail:
                debug_text = f"{debug_text}\n\nLLM trace (tail):\n{tail}"
        return [], {}, health_status, "", None, health_status, debug_text

    state: Dict[str, Any] = {
        "pdf_name": stored_path,
        "user_input": _safe_str(instruction),
        "default_instrument": _safe_str(default_instrument),
        "overrides": {},
        "debug": bool(debug),
    }

    try:
        extracted = order_graph.extract_songs_node(state)
    except Exception as exc:
        LOGGER.exception("Song extraction failed")
        if debug_text:
            tail = _read_llm_trace_tail()
            if tail:
                debug_text = f"{debug_text}\n\nLLM trace (tail):\n{tail}"
        return [], {}, f"Song extraction failed: {exc}", "", None, health_status, debug_text

    songs = extracted.get("songs", {}) or {}
    rows = _build_song_rows(songs, _safe_str(default_instrument))
    run_id = extracted.get("run_id") or uuid.uuid4().hex[:10]
    order_folder = extracted.get("order_folder") or "unknown_date"

    status = f"Found {len(rows)} song(s)."
    if not rows:
        _, errs = _summarize_errors(extracted, 0)
        if errs:
            status = f"No songs found. Last error: {errs[-1]}"

    order_state = {
        "pdf_name": stored_path,
        "order_folder": order_folder,
        "run_id": run_id,
        "default_instrument": _safe_str(default_instrument),
        "user_input": _safe_str(instruction),
        "debug": bool(debug),
        "ollama_url": url,
        "ollama_model": model,
        "ollama_host": host,
        "ollama_models_path": models_path,
    }

    if debug_text:
        tail = _read_llm_trace_tail()
        if tail:
            debug_text = f"{debug_text}\n\nLLM trace (tail):\n{tail}"
    return rows, order_state, status, "", None, health_status, debug_text


def run_processing(
    rows: List[List[Any]] | None,
    order_state: Dict[str, Any] | None,
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    parallel_processing: bool,
    max_procs: int,
    top_n: int,
    debug: bool,
    save_screens: bool,
    save_html: bool,
    save_errors_only: bool,
    preview_enabled: bool,
    progress: gr.Progress = gr.Progress(),  # type: ignore[assignment]
) -> Iterable[Tuple[str, str | None, List[Tuple[str, str]]]]:
    """Process verified songs and stream progress updates."""
    log_lines: List[str] = []
    preview_items: List[Tuple[str, str]] = []
    if not order_state or not order_state.get("pdf_name"):
        log_lines.append("No order loaded. Run analysis first.")
        yield "\n".join(log_lines), None, preview_items
        return

    default_instrument = _safe_str(order_state.get("default_instrument"))
    songs = _parse_song_rows(rows, default_instrument)
    if not songs:
        log_lines.append("No songs selected for processing.")
        yield "\n".join(log_lines), None, preview_items
        return

    cancel_event = _new_cancel_event(bool(parallel_processing))
    try:
        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None, preview_items
            return

        effective_save_screens = bool(save_screens or preview_enabled)
        effective_errors_only = False if preview_enabled else bool(save_errors_only)
        _apply_capture_env(effective_save_screens, save_html, effective_errors_only)

        url, model, _, _ = _apply_ollama_env(
            ollama_url,
            ollama_model,
            ollama_host,
            ollama_models_path,
        )
        if ollama_debug:
            os.environ["WMRA_LLM_TRACE"] = "1"
            os.environ["WMRA_LLM_TRACE_VERBOSE"] = "1"
        else:
            os.environ.pop("WMRA_LLM_TRACE", None)
            os.environ.pop("WMRA_LLM_TRACE_VERBOSE", None)
        diag = ping_ollama(url, model)
        if not diag.get("ok", False) or not diag.get("has_model", False):
            log_lines.append(f"WARNING: {_format_ollama_status(diag)}")
            yield "\n".join(log_lines), None, preview_items
        elif ollama_debug:
            log_lines.append("Ollama debug enabled; click Check Ollama for details.")
            yield "\n".join(log_lines), None, preview_items

        run_id = _safe_str(order_state.get("run_id")) or uuid.uuid4().hex[:10]
        order_folder = _safe_str(order_state.get("order_folder")) or "unknown_date"
        preview_root = ""
        if preview_enabled:
            preview_root = os.path.join(os.getcwd(), "output", "previews", order_folder, run_id)
            try:
                os.makedirs(preview_root, exist_ok=True)
            except Exception:
                preview_root = ""
        total = len(songs)
        log_lines.append(f"Processing {total} song(s).")
        progress(0.0, desc="Starting")
        yield "\n".join(log_lines), None, preview_items

        def _apply_progress(stage: str, done: int, total_steps: int) -> None:
            if total_steps <= 0:
                return
            ratio = max(0.0, min(1.0, done / total_steps))
            if stage == "scrape":
                desc = f"Scraping ({done}/{total_steps})"
            elif stage == "watermark":
                desc = f"Removing watermark ({done}/{total_steps})"
            elif stage == "upscale":
                desc = f"Upscaling ({done}/{total_steps})"
            else:
                desc = f"{stage} ({done}/{total_steps})"
            progress(ratio, desc=desc)

        state: Dict[str, Any] = {
            "pdf_name": order_state.get("pdf_name"),
            "order_folder": order_folder,
            "user_input": _safe_str(order_state.get("user_input")),
            "default_instrument": default_instrument,
            "songs": songs,
            "top_n": int(top_n) if top_n else 3,
            "debug": bool(debug),
            "run_id": run_id,
            "_cancel_event": cancel_event,
        }
        state["parallel_scrape"] = bool(parallel_processing)
        if preview_root:
            state["preview_root"] = preview_root

        if parallel_processing:
            max_procs_val = int(max_procs) if max_procs else 0
            if max_procs_val > 0:
                os.environ["ORDER_MAX_PROCS"] = str(max_procs_val)
            else:
                os.environ.pop("ORDER_MAX_PROCS", None)
            log_lines.append(
                f"Parallel scraping enabled (max_procs={max_procs_val or 'all'})."
            )
            progress(0.0, desc=f"Scraping (0/{total})")
            yield "\n".join(log_lines), None, preview_items
        else:
            progress(0.0, desc=f"Scraping (0/{total})")
            log_lines.append("Scraping all songs before processing.")
            yield "\n".join(log_lines), None, preview_items

        def _run_process(progress_cb_fn):
            if parallel_processing:
                return order_graph.process_songs_parallel_node(state, progress_cb=progress_cb_fn)
            return order_graph.process_songs_node(state, progress_cb=progress_cb_fn)

        if preview_enabled:
            event_queue: queue.Queue[Tuple[str, int, int]] = queue.Queue()

            def progress_cb(stage: str, done: int, total_steps: int) -> None:
                event_queue.put((stage, done, total_steps))

            def _apply_events() -> bool:
                changed = False
                while True:
                    try:
                        stage, done, total_steps = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    _apply_progress(stage, done, total_steps)
                    changed = True
                return changed

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_process, progress_cb)
                last_preview: List[Tuple[str, str]] = []
                last_yield = 0.0
                while True:
                    progress_changed = _apply_events()
                    preview_changed = False
                    if preview_root:
                        preview_items = _collect_preview_images(preview_root)
                        if preview_items != last_preview:
                            last_preview = list(preview_items)
                            preview_changed = True
                    now = time.time()
                    if progress_changed or preview_changed or (now - last_yield) > 2.0:
                        yield "\n".join(log_lines), None, preview_items
                        last_yield = now
                    if future.done():
                        break
                    time.sleep(0.2)
                _apply_events()
                if preview_root:
                    preview_items = _collect_preview_images(preview_root)
                try:
                    state = future.result()
                except Exception as exc:
                    LOGGER.exception("Batch processing failed")
                    log_lines.append(f"Batch processing failed: {exc}")
                    yield "\n".join(log_lines), None, preview_items
                    return
        else:
            def progress_cb(stage: str, done: int, total_steps: int) -> None:
                _apply_progress(stage, done, total_steps)

            try:
                state = _run_process(progress_cb)
            except Exception as exc:
                LOGGER.exception("Batch processing failed")
                log_lines.append(f"Batch processing failed: {exc}")
                yield "\n".join(log_lines), None, preview_items
                return

        if state.get("cancelled"):
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None, preview_items
            return
        _, new_errs = _summarize_errors(state, 0)
        if new_errs:
            log_lines.extend(new_errs)
            yield "\n".join(log_lines), None, preview_items

        final_pdfs = list(state.get("final_pdfs", []) or [])
        success_count = sum(1 for p in final_pdfs if p)
        log_lines.append(f"Completed {success_count} PDF(s).")
        if success_count == 0:
            _, new_errs = _summarize_errors(state, 0)
            if not new_errs:
                log_lines.append("No PDFs generated; check output logs for scraper errors.")

        zip_path = _build_zip(order_folder, run_id, final_pdfs)
        if zip_path:
            log_lines.append(f"Zip ready: {zip_path}")
        else:
            log_lines.append("Zip could not be created; no output files found.")
        yield "\n".join(log_lines), zip_path, preview_items
    finally:
        _set_cancel_event(None)


def run_single_song(
    prompt: str,
    title_override: str,
    artist_override: str,
    key_override: str,
    instrument_override: str,
    default_instrument: str,
    ollama_url: str,
    ollama_model: str,
    ollama_host: str,
    ollama_models_path: str,
    ollama_debug: bool,
    debug: bool,
    save_screens: bool,
    save_html: bool,
    save_errors_only: bool,
    progress: gr.Progress = gr.Progress(),  # type: ignore[assignment]
) -> Iterable[Tuple[str, str | None]]:
    """Run the single-song scrape -> watermark -> upscale -> PDF pipeline."""
    log_lines: List[str] = []
    if not _safe_str(prompt) and not _safe_str(title_override):
        log_lines.append("Provide a prompt or a title to begin.")
        yield "\n".join(log_lines), None
        return

    _new_cancel_event(False)
    try:
        run_ts = _new_run_ts()
        _apply_capture_env(save_screens, save_html, save_errors_only)
        url, model, _, _ = _apply_ollama_env(
            ollama_url,
            ollama_model,
            ollama_host,
            ollama_models_path,
        )
        if ollama_debug:
            os.environ["WMRA_LLM_TRACE"] = "1"
            os.environ["WMRA_LLM_TRACE_VERBOSE"] = "1"
        else:
            os.environ.pop("WMRA_LLM_TRACE", None)
            os.environ.pop("WMRA_LLM_TRACE_VERBOSE", None)
        diag = ping_ollama(url, model)
        if not diag.get("ok", False) or not diag.get("has_model", False):
            log_lines.append(f"WARNING: {_format_ollama_status(diag)}")
            yield "\n".join(log_lines), None

        title = _safe_str(title_override)
        artist = _safe_str(artist_override)
        instrument = _safe_str(instrument_override)
        key = _safe_str(key_override)
        prompt_text = _safe_str(prompt)

        if (not title or not instrument or not key) and prompt_text:
            if not diag.get("ok", False) or not diag.get("has_model", False):
                log_lines.append("LLM unavailable; using heuristic parsing for missing fields.")
                yield "\n".join(log_lines), None
            parse_state = {
                "user_input": prompt_text,
                "title": title,
                "instrument": instrument,
                "key": key,
            }
            try:
                parsed = single_song_graph.parser_node(parse_state)
            except Exception as exc:
                LOGGER.warning("Single-song parser failed: %s", exc)
                parsed = {}
                if debug:
                    log_lines.append(f"Parser error: {exc}")
                    yield "\n".join(log_lines), None
            if isinstance(parsed, dict):
                title = _safe_str(parsed.get("title") or title)
                instrument = _safe_str(parsed.get("instrument") or instrument)
                key = _safe_str(parsed.get("key") or key)
            if (not title or not instrument or not key) and prompt_text:
                try:
                    fallback = single_song_graph._heuristic_parse(prompt_text)
                except Exception:
                    fallback = {}
                if isinstance(fallback, dict):
                    if not title:
                        title = _safe_str(fallback.get("title"))
                    if not instrument:
                        instrument = _safe_str(fallback.get("instrument"))
                    if not key:
                        key = _safe_str(fallback.get("key"))

        if not instrument:
            instrument = _safe_str(default_instrument)
        if not title:
            log_lines.append("Missing title. Add it to the prompt or Title field.")
            yield "\n".join(log_lines), None
            return
        if not key:
            log_lines.append("Missing key. Add it to the prompt (e.g. 'in C') or Key field.")
            yield "\n".join(log_lines), None
            return

        log_lines.append(
            f"Using title='{title}', artist='{artist or ''}', key='{key}', instrument='{instrument}'."
        )
        progress(0.1, desc="Scraping")
        log_lines.append("Scraping...")
        yield "\n".join(log_lines), None
        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return

        try:
            scraped = scrape_music.invoke(
                {
                    "title": title,
                    "instrument": instrument,
                    "key": key,
                    "artist": artist or None,
                    "top_n": 1,
                }
            )
        except Exception as exc:
            LOGGER.exception("Single-song scrape failed")
            log_lines.append(f"Scrape failed: {exc}")
            yield "\n".join(log_lines), None
            return

        image_dir: str | None = None
        if isinstance(scraped, list):
            if scraped and isinstance(scraped[0], dict):
                existing_pdf = scraped[0].get("existing_pdf")
                if existing_pdf:
                    log_lines.append(f"Existing PDF found: {existing_pdf}")
                    yield "\n".join(log_lines), existing_pdf
                    return
                image_dir = scraped[0].get("image_dir") or scraped[0].get("tmp_dir")
        elif isinstance(scraped, str):
            image_dir = scraped

        if not image_dir:
            log_lines.append("Scraper returned no images.")
            yield "\n".join(log_lines), None
            return

        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        progress(0.4, desc="Removing watermark")
        log_lines.append("Removing watermark...")
        yield "\n".join(log_lines), None
        try:
            processed_dir = remove_watermark.invoke({"input_dir": image_dir})
        except Exception as exc:
            LOGGER.exception("Watermark removal failed")
            log_lines.append(f"Watermark removal failed: {exc}")
            yield "\n".join(log_lines), None
            return

        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        progress(0.7, desc="Upscaling")
        log_lines.append("Upscaling...")
        yield "\n".join(log_lines), None
        try:
            upscaled_dir = upscale_images.invoke({"input_dir": processed_dir})
        except Exception as exc:
            LOGGER.exception("Upscaling failed")
            log_lines.append(f"Upscaling failed: {exc}")
            yield "\n".join(log_lines), None
            return

        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        progress(0.9, desc="Assembling PDF")
        log_lines.append("Assembling PDF...")
        yield "\n".join(log_lines), None
        meta = {
            "title": title,
            "artist": artist or "",
            "key": key,
            "instrument": instrument,
            "run_ts": run_ts,
        }
        try:
            pdf_path = assemble_pdf.invoke({"image_dir": upscaled_dir, "meta": meta})
        except Exception as exc:
            LOGGER.exception("PDF assembly failed")
            log_lines.append(f"PDF assembly failed: {exc}")
            yield "\n".join(log_lines), None
            return

        log_lines.append(f"PDF ready: {pdf_path}")
        yield "\n".join(log_lines), pdf_path
    finally:
        _set_cancel_event(None)


def run_manual_images(
    image_files: Any,
    title: str,
    artist: str,
    key: str,
    instrument: str,
    default_instrument: str,
    remove_first: bool,
    upscale_second: bool,
    debug: bool,
    progress: gr.Progress = gr.Progress(),  # type: ignore[assignment]
) -> Iterable[Tuple[str, str | None]]:
    """Process uploaded images through watermark removal/upscaling and assemble a PDF."""
    log_lines: List[str] = []
    files = _coerce_file_list(image_files)
    if not files:
        log_lines.append("Upload one or more images to begin.")
        yield "\n".join(log_lines), None
        return

    _new_cancel_event(False)
    try:
        run_ts = _new_run_ts()
        try:
            image_dir = _store_image_uploads(files, run_ts)
        except Exception as exc:
            log_lines.append(f"Failed to store uploaded images: {exc}")
            yield "\n".join(log_lines), None
            return

        if debug:
            log_lines.append(f"Stored {len(files)} image(s) in {image_dir}.")
        else:
            log_lines.append(f"Stored {len(files)} image(s).")
        yield "\n".join(log_lines), None

        current_dir = image_dir
        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        if remove_first:
            progress(0.3, desc="Removing watermark")
            log_lines.append("Removing watermark...")
            yield "\n".join(log_lines), None
            try:
                current_dir = remove_watermark.invoke({"input_dir": current_dir})
            except Exception as exc:
                LOGGER.exception("Manual watermark removal failed")
                log_lines.append(f"Watermark removal failed: {exc}")
                yield "\n".join(log_lines), None
                return

        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        if upscale_second:
            progress(0.6, desc="Upscaling")
            log_lines.append("Upscaling...")
            yield "\n".join(log_lines), None
            try:
                current_dir = upscale_images.invoke({"input_dir": current_dir})
            except Exception as exc:
                LOGGER.exception("Manual upscaling failed")
                log_lines.append(f"Upscaling failed: {exc}")
                yield "\n".join(log_lines), None
                return

        if _cancel_requested():
            log_lines.append("Cancelled by user.")
            yield "\n".join(log_lines), None
            return
        progress(0.9, desc="Assembling PDF")
        log_lines.append("Assembling PDF...")
        yield "\n".join(log_lines), None

        title_val = _safe_str(title) or "manual_upload"
        artist_val = _safe_str(artist)
        key_val = _safe_str(key)
        instrument_val = _safe_str(instrument) or _safe_str(default_instrument)
        meta = {
            "title": title_val,
            "artist": artist_val,
            "key": key_val,
            "instrument": instrument_val,
            "run_ts": run_ts,
        }
        try:
            pdf_path = assemble_pdf.invoke({"image_dir": current_dir, "meta": meta})
        except Exception as exc:
            LOGGER.exception("Manual PDF assembly failed")
            log_lines.append(f"PDF assembly failed: {exc}")
            yield "\n".join(log_lines), None
            return

        log_lines.append(f"PDF ready: {pdf_path}")
        yield "\n".join(log_lines), pdf_path
    finally:
        _set_cancel_event(None)


def build_app() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(title="Order of Worship GUI") as demo:
        gr.Markdown(
            "# Order of Worship GUI\n"
            "Upload a service order, extract songs, confirm selections, then run the pipeline."
        )

        initial_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        initial_host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        initial_models_path = _safe_str(os.environ.get("OLLAMA_MODELS")) or _detect_ollama_models_path()
        initial_models: List[str] = [MODEL_PLACEHOLDER]
        initial_model = MODEL_PLACEHOLDER

        with gr.Accordion("Ollama & Global Settings", open=False):
            with gr.Row():
                ollama_url = gr.Textbox(
                    label="OLLAMA_URL",
                    value=initial_url,
                )
                ollama_model = gr.Dropdown(
                    label="OLLAMA_MODEL",
                    choices=initial_models,
                    value=initial_model,
                    allow_custom_value=False,
                )

            with gr.Row():
                ollama_host = gr.Textbox(
                    label="OLLAMA_HOST",
                    value=initial_host,
                    placeholder="127.0.0.1:11434",
                )
                ollama_models_path = gr.Textbox(
                    label="OLLAMA_MODELS (WSL path)",
                    value=initial_models_path,
                    placeholder="/mnt/c/Users/<you>/AppData/Local/Ollama/models",
                )

            gr.Markdown(
                "Tip: `OLLAMA_MODELS` should point to the Windows models folder using a WSL path."
            )

            with gr.Row():
                check_btn = gr.Button("Check Ollama")
                start_btn = gr.Button("Start Ollama (WSL)")
                stop_btn = gr.Button("Stop Ollama (WSL)")
                restart_btn = gr.Button("Force Restart (WSL)")

            ollama_status = gr.Markdown()
            ollama_debug_out = gr.Textbox(
                label="Ollama debug output",
                lines=10,
                interactive=False,
            )

            with gr.Row():
                default_instrument = gr.Textbox(
                    label="Default instrument",
                    value="French Horn in F",
                )
                debug = gr.Checkbox(label="Debug logging", value=False)
                ollama_debug = gr.Checkbox(label="Ollama debug", value=False)
            with gr.Row():
                save_screens = gr.Checkbox(label="Save screenshots", value=False)
                save_html = gr.Checkbox(label="Save HTML", value=False)
                save_errors_only = gr.Checkbox(label="Only save on errors", value=True)
            with gr.Row():
                preview_enabled = gr.Checkbox(
                    label="Show Selenium preview (snapshots)",
                    value=False,
                )

        with gr.Tabs():
            with gr.TabItem("Order of Worship"):
                gr.Markdown(
                    "Upload a service order, extract songs, confirm selections, then run the pipeline."
                )
                with gr.Row():
                    pdf_file = gr.File(
                        label="Order of Worship PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    instruction = gr.Textbox(
                        label="Instruction / prompt (used for extraction)",
                        value="Download the songs for the French Horn.",
                        lines=4,
                    )

                with gr.Row():
                    top_n = gr.Slider(
                        1,
                        5,
                        value=3,
                        step=1,
                        label="Max candidates per song",
                    )
                    parallel_processing = gr.Checkbox(
                        label="Parallel scraping (multi-process)",
                        value=True,
                    )
                    max_procs = gr.Number(
                        label="Max parallel processes (0 = all)",
                        value=2,
                        precision=0,
                    )

                analyze_btn = gr.Button("Analyze order")
                status = gr.Markdown()

                songs_table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=TABLE_TYPES,
                    row_count=(1, "dynamic"),
                    col_count=(len(TABLE_HEADERS), "fixed"),
                    interactive=True,
                    label="Detected songs (edit before running)",
                )

                run_btn = gr.Button("Run pipeline")
                cancel_btn = gr.Button("Cancel run")
                progress_log = gr.Textbox(
                    label="Progress",
                    lines=12,
                    interactive=False,
                )
                preview_gallery = gr.Gallery(
                    label="Selenium preview",
                    columns=2,
                    visible=False,
                )
                output_zip = gr.File(label="Download zip")

            with gr.TabItem("Single Song"):
                gr.Markdown(
                    "Enter a prompt or fill in fields to scrape and process a single song."
                )
                single_prompt = gr.Textbox(
                    label="Single song prompt",
                    lines=2,
                    placeholder="Download 'Fur Elise' for French Horn in F",
                )
                with gr.Row():
                    single_title = gr.Textbox(label="Title (optional)")
                    single_artist = gr.Textbox(label="Artist (optional)")
                    single_key = gr.Textbox(label="Key (optional)")
                    single_instrument = gr.Textbox(
                        label="Instrument (optional)",
                        placeholder="Leave blank to use the default instrument",
                    )
                with gr.Row():
                    single_run_btn = gr.Button("Run single song")
                    single_cancel_btn = gr.Button("Cancel single song")
                single_log = gr.Textbox(
                    label="Single song progress",
                    lines=8,
                    interactive=False,
                )
                single_pdf = gr.File(label="Single song PDF")

            with gr.TabItem("Manual Images"):
                gr.Markdown(
                    "Upload images and optionally remove watermarks/upscale before PDF assembly."
                )
                manual_images = gr.File(
                    label="Images",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    file_count="multiple",
                    type="filepath",
                )
                with gr.Row():
                    manual_title = gr.Textbox(label="Title (optional)")
                    manual_artist = gr.Textbox(label="Artist (optional)")
                    manual_key = gr.Textbox(label="Key (optional)")
                    manual_instrument = gr.Textbox(
                        label="Instrument (optional)",
                        placeholder="Leave blank to use the default instrument",
                    )
                with gr.Row():
                    manual_remove = gr.Checkbox(label="Remove watermark", value=True)
                    manual_upscale = gr.Checkbox(label="Upscale", value=True)
                with gr.Row():
                    manual_run_btn = gr.Button("Process images")
                    manual_cancel_btn = gr.Button("Cancel image processing")
                manual_log = gr.Textbox(
                    label="Manual image progress",
                    lines=8,
                    interactive=False,
                )
                manual_pdf = gr.File(label="Manual images PDF")

        order_state = gr.State({})
        ollama_proc_state = gr.State({})

        analyze_btn.click(
            analyze_order,
            inputs=[
                pdf_file,
                instruction,
                default_instrument,
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                debug,
            ],
            outputs=[
                songs_table,
                order_state,
                status,
                progress_log,
                output_zip,
                ollama_status,
                ollama_debug_out,
            ],
        )

        run_btn.click(
            run_processing,
            inputs=[
                songs_table,
                order_state,
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                parallel_processing,
                max_procs,
                top_n,
                debug,
                save_screens,
                save_html,
                save_errors_only,
                preview_enabled,
            ],
            outputs=[progress_log, output_zip, preview_gallery],
        )

        cancel_btn.click(
            cancel_order_processing,
            outputs=[progress_log, output_zip, preview_gallery],
        )

        preview_enabled.change(
            lambda show: gr.update(visible=bool(show)),
            inputs=[preview_enabled],
            outputs=[preview_gallery],
        )

        check_btn.click(
            check_ollama,
            inputs=[
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                ollama_proc_state,
            ],
            outputs=[ollama_status, ollama_debug_out, ollama_model],
        )

        start_btn.click(
            start_ollama_server,
            inputs=[
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                ollama_proc_state,
            ],
            outputs=[ollama_proc_state, ollama_status, ollama_debug_out],
        )

        stop_btn.click(
            stop_ollama_server,
            inputs=[
                ollama_proc_state,
                ollama_debug,
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
            ],
            outputs=[ollama_proc_state, ollama_status, ollama_debug_out],
        )

        restart_btn.click(
            force_restart_ollama,
            inputs=[
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                ollama_proc_state,
            ],
            outputs=[ollama_proc_state, ollama_status, ollama_debug_out],
        )

        single_run_btn.click(
            run_single_song,
            inputs=[
                single_prompt,
                single_title,
                single_artist,
                single_key,
                single_instrument,
                default_instrument,
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                debug,
                save_screens,
                save_html,
                save_errors_only,
            ],
            outputs=[single_log, single_pdf],
        )

        single_cancel_btn.click(
            cancel_processing,
            outputs=[single_log, single_pdf],
        )

        manual_run_btn.click(
            run_manual_images,
            inputs=[
                manual_images,
                manual_title,
                manual_artist,
                manual_key,
                manual_instrument,
                default_instrument,
                manual_remove,
                manual_upscale,
                debug,
            ],
            outputs=[manual_log, manual_pdf],
        )

        manual_cancel_btn.click(
            cancel_processing,
            outputs=[manual_log, manual_pdf],
        )

        demo.load(
            auto_start_ollama,
            inputs=[
                ollama_url,
                ollama_model,
                ollama_host,
                ollama_models_path,
                ollama_debug,
                ollama_proc_state,
            ],
            outputs=[ollama_proc_state, ollama_status, ollama_debug_out, ollama_model],
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = True) -> None:
    """Launch the GUI server."""
    app = build_app()
    app.queue()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        allowed_paths=[str(Path(os.getcwd()) / "output")],
    )


def main() -> None:
    """Command-line entry point for the GUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Order of Worship GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
