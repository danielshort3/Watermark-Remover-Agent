"""
Helpers for invoking the Ollama-powered chat agent.

This module exposes a convenience wrapper that now prefers running the
LangChain-based agent (constructed via :func:`get_ollama_agent`) so the LLM can
plan and execute tools autonomously.  If the agent cannot be constructed (for
example, when optional dependencies are missing) the helper falls back to a
direct ChatOllama invocation, matching the previous behaviour.
"""

from __future__ import annotations

import logging
import json
import os
import socket
import struct
import time
import urllib.request
from typing import Any, Dict, Optional
from config.settings import DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL


def _is_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except Exception:
        return False


def _get_wsl_gateway_ip() -> str | None:
    """Return the WSL default gateway IP, if available."""
    if not _is_wsl():
        return None
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as handle:
            for line in handle.readlines()[1:]:
                fields = line.strip().split()
                if len(fields) < 4:
                    continue
                destination, gateway, flags = fields[1], fields[2], fields[3]
                if destination != "00000000":
                    continue
                if int(flags, 16) & 2 == 0:
                    continue
                ip = socket.inet_ntoa(struct.pack("<L", int(gateway, 16)))
                if ip and ip != "0.0.0.0":
                    return ip
    except Exception:
        return None
    return None


def _candidate_ollama_urls(base_url: str) -> list[str]:
    urls = [base_url]
    norm = (
        base_url.replace("127.0.0.1", "localhost")
        .replace("0.0.0.0", "localhost")
    )
    if "localhost" in norm:
        alt = norm.replace("localhost", "host.docker.internal")
        if alt not in urls:
            urls.append(alt)
        wsl_gateway = _get_wsl_gateway_ip()
        if wsl_gateway:
            wsl_url = norm.replace("localhost", wsl_gateway)
            if wsl_url not in urls:
                urls.append(wsl_url)
    return urls


def _ping_ollama(base_url: str, model: str) -> Dict[str, Any]:
    """Ping an Ollama server for diagnostics.

    Contacts the Ollama API at ``base_url`` to retrieve version and tag
    information and checks whether the specified ``model`` is available.
    Returns a dictionary with keys ``ok`` (bool), ``version`` (dict or
    None), ``models`` (list of model names) and ``has_model`` (bool)
    along with the original ``base_url`` and ``model`` values.  On
    failure it sets ``ok=False`` and includes an ``error`` string.
    """
    errors: list[str] = []
    for candidate in _candidate_ollama_urls(base_url):
        out: Dict[str, Any] = {"ok": True, "base_url": candidate, "model": model}
        try:
            with urllib.request.urlopen(f"{candidate.rstrip('/')}/api/version", timeout=5) as resp:
                out["version"] = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            errors.append(f"Failed to reach {candidate}/api/version: {exc}")
            continue
        try:
            with urllib.request.urlopen(f"{candidate.rstrip('/')}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8")).get("models", [])
                names = [m.get("name") or m.get("model") for m in tags]
                out["models"] = names
                out["has_model"] = model in names
        except Exception as exc:
            errors.append(f"Failed to query tags at {candidate}: {exc}")
            continue
        return out
    # Exhausted all candidates
    return {"ok": False, "base_url": base_url, "model": model, "error": "; ".join(errors)}


try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore

try:
    from watermark_remover.agent.ollama_agent import get_ollama_agent  # type: ignore
    _AGENT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import-time diagnostics
    get_ollama_agent = None  # type: ignore
    _AGENT_IMPORT_ERROR = exc

_LLM: Any = None
_LLM_MODEL: Optional[str] = None
_LLM_BASE_URL: Optional[str] = None
_AGENT: Any = None
_AGENT_MODEL: Optional[str] = None
_AGENT_BASE_URL: Optional[str] = None

_logger = logging.getLogger(__name__)


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _llm_trace_enabled() -> bool:
    return _env_truthy(os.environ.get("WMRA_LLM_TRACE"))


def _llm_trace_verbose() -> bool:
    return _env_truthy(os.environ.get("WMRA_LLM_TRACE_VERBOSE"))


def _llm_trace_path() -> str:
    path = os.environ.get("WMRA_LLM_TRACE_PATH")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return path
    base = os.environ.get("WMRA_LOG_DIR")
    if not base:
        run_ts = os.environ.get("RUN_TS")
        if not run_ts:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            os.environ.setdefault("RUN_TS", run_ts)
        base = os.path.join(os.getcwd(), "output", "logs", run_ts)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "llm_trace.jsonl")


def _trace_event(payload: Dict[str, Any]) -> None:
    if not _llm_trace_enabled():
        return
    event = dict(payload)
    event.setdefault("ts", time.time())
    label = os.environ.get("WMRA_LLM_TRACE_LABEL")
    if label and "label" not in event:
        event["label"] = label
    prompt_path = os.environ.get("WMRA_LLM_TRACE_PROMPT_PATH")
    if prompt_path:
        event.setdefault("prompt_path", prompt_path)
    try:
        path = _llm_trace_path()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _get_llm(base_url: str, model: str) -> Any:
    if ChatOllama is None:
        raise ImportError("ChatOllama is not available; install langchain-ollama.")
    global _LLM, _LLM_MODEL, _LLM_BASE_URL
    if _LLM is not None and _LLM_MODEL == model and _LLM_BASE_URL == base_url:
        return _LLM
    _LLM = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
        keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
    )
    _LLM_MODEL = model
    _LLM_BASE_URL = base_url
    return _LLM


def _get_agent(base_url: str, model: str) -> Any:
    """Return or construct a cached LangChain agent for the given endpoint."""
    if get_ollama_agent is None:
        raise ImportError(
            "The LangChain agent dependencies are not installed; install "
            "langchain, langchain-community, and langchain-ollama to enable "
            "agent execution."
            + (
                f" (import error: {_AGENT_IMPORT_ERROR})"
                if _AGENT_IMPORT_ERROR is not None
                else ""
            )
        )
    global _AGENT, _AGENT_MODEL, _AGENT_BASE_URL
    if _AGENT is not None and _AGENT_MODEL == model and _AGENT_BASE_URL == base_url:
        return _AGENT
    agent = get_ollama_agent(model_name=model, base_url=base_url)
    _AGENT = agent
    _AGENT_MODEL = model
    _AGENT_BASE_URL = base_url
    return agent


def _extract_agent_output(result: Any) -> str:
    """Normalise the agent response into a printable string."""
    if isinstance(result, dict):
        for key in ("output", "result", "text", "content"):
            value = result.get(key)
            if value is not None:
                return str(value)
        return str(result)
    if hasattr(result, "content"):
        return str(result.content)  # type: ignore[attr-defined]
    return str(result)


def run_instruction(instruction: str) -> str:
    """Execute a task using the Ollama‑powered LangChain agent.

    This convenience wrapper constructs and invokes a LangChain agent on
    demand.  It first ensures a non‑empty prompt, then verifies that the
    Ollama server specified by the ``OLLAMA_URL`` environment variable
    (defaulting to ``http://localhost:11434``) is reachable and that
    the model specified by ``OLLAMA_MODEL`` (defaulting to ``qwen3:8b``)
    is available.  It lazily imports the agent factory function
    :func:`get_ollama_agent` and returns the agent's response as a
    string.  Any error encountered is returned as a descriptive string.

    Parameters
    ----------
    instruction : str
        A natural‑language instruction to pass to the agent.

    Returns
    -------
    str
        The agent's response or an error message.
    """
    user_prompt = (instruction or "").strip()
    if not user_prompt:
        _trace_event({"event": "run_instruction_empty"})
        return (
            "No instruction provided. Please supply a natural‑language task "
            "for the agent to perform."
        )
    prompt = user_prompt
    lowered = user_prompt.lower()
    if ".pdf" in lowered and ("order of worship" in lowered or "order" in lowered):
        prompt = (
            user_prompt
            + "\n\nReminder: When processing an order-of-worship PDF you MUST call the "
              "ensure_order_pdf tool to copy the source PDF into the output/orders/<date>/ "
              "directory using a '00_' prefix (e.g. '00_October_12_2025_Order_Of_Worship.pdf')."
        )
    elif ".pdf" not in lowered and any(
        keyword in lowered for keyword in ("download", "find", "locate", "fetch")
    ):
        prompt = (
            user_prompt
            + "\n\nReminder: When the user requests a specific song (not an order PDF), "
              "use the scrape_music tool with the requested title/instrument/key, then run "
              "remove_watermark, upscale_images, and assemble_pdf for that song only."
        )
    base_url_env = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    start_ts = time.perf_counter()
    trace_payload: Dict[str, Any] = {
        "event": "run_instruction_start",
        "prompt_chars": len(prompt),
    }
    if _llm_trace_verbose():
        trace_payload["prompt_preview"] = prompt[:500]
    _trace_event(trace_payload)

    ping_start = time.perf_counter()
    diag = _ping_ollama(base_url_env, model)
    _trace_event(
        {
            "event": "ollama_ping",
            "duration_ms": round((time.perf_counter() - ping_start) * 1000, 2),
            "ok": bool(diag.get("ok")),
            "has_model": bool(diag.get("has_model")),
            "base_url": diag.get("base_url"),
            "model": diag.get("model"),
        }
    )
    if not diag.get("ok", False):
        _trace_event(
            {
                "event": "run_instruction_end",
                "duration_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                "error": "ollama_unreachable",
            }
        )
        return f"Cannot reach Ollama at {base_url_env}: {diag.get('error', 'Unknown error')}"
    resolved_base_url = str(diag.get("base_url") or base_url_env)
    if not diag.get("has_model", False):
        _trace_event(
            {
                "event": "run_instruction_end",
                "duration_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                "error": "model_not_found",
                "model": model,
            }
        )
        return (
            f"Model '{model}' not found on Ollama server. Available models: "
            f"{diag.get('models')}"
        )

    agent: Any | None = None
    agent_error: Optional[Exception] = None
    try:
        agent = _get_agent(resolved_base_url, model)
    except Exception as exc:  # noqa: BLE001 - Return message downstream
        agent_error = exc
        _trace_event(
            {
                "event": "agent_unavailable",
                "error": str(exc),
            }
        )

    answer: str
    if agent is not None and agent_error is None:
        try:
            agent_start = time.perf_counter()
            result = agent.invoke({"input": prompt})
            answer = _extract_agent_output(result)
            _trace_event(
                {
                    "event": "agent_invoke",
                    "duration_ms": round((time.perf_counter() - agent_start) * 1000, 2),
                    "answer_chars": len(answer),
                }
            )
        except Exception as exc:  # noqa: BLE001
            agent_error = exc
            _trace_event(
                {
                    "event": "agent_invoke_error",
                    "error": str(exc),
                }
            )
        else:
            agent_error = None

    if agent_error is not None or agent is None:
        # Fall back to the direct model call when the agent is unavailable.
        if agent_error is not None:
            _logger.warning(
                "Falling back to direct LLM call because the agent is unavailable: %s",
                agent_error,
            )
        try:
            llm_start = time.perf_counter()
            llm = _get_llm(resolved_base_url, model)
            response = llm.invoke(prompt)  # type: ignore
            answer = (
                response.content if hasattr(response, "content") else str(response)
            )
            _trace_event(
                {
                    "event": "direct_llm_invoke",
                    "duration_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                    "answer_chars": len(answer),
                }
            )
        except Exception as exc:  # noqa: BLE001
            _trace_event(
                {
                    "event": "direct_llm_error",
                    "error": str(exc),
                }
            )
            return f"Error executing instruction: {exc}"

    try:
        # Persist the thought process and steps to a log file under the output directory.
        # Persist the thought process to a log file under the run‑specific
        # logs directory.  Use the WMRA_LOG_DIR environment variable if set.
        log_root = os.environ.get("WMRA_LOG_DIR") or os.path.join(os.getcwd(), "logs")
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(log_root, "thoughts_and_steps.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Instruction: {prompt}\n")
            f.write(str(answer) + "\n\n")
    except Exception:
        pass
    # Also log the model's thought process to the main pipeline logger so
    # users can inspect why the agent made certain decisions (e.g. choosing
    # a different key when the requested one is unavailable).  We strip
    # newline characters as the CSV formatter will sanitise them.
    try:
        from watermark_remover.agent.tools import logger  # type: ignore
        # Use extra to ensure CSV columns align
        logger.info(
            f"AGENT_THOUGHTS: {str(answer).strip()}",
            extra={"button_text": "", "xpath": "", "url": "", "screenshot": ""},
        )
    except Exception:
        pass
    _trace_event(
        {
            "event": "run_instruction_end",
            "duration_ms": round((time.perf_counter() - start_ts) * 1000, 2),
            "answer_chars": len(str(answer)),
            "agent_used": agent is not None and agent_error is None,
        }
    )
    return str(answer)
