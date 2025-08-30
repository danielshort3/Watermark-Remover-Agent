"""
Helpers for invoking the Ollama‑powered chat agent.

This module provides a direct interface to execute natural‑language
instructions using a single ChatOllama LLM (no LangChain AgentExecutor).
It avoids deprecated Agent APIs and keeps calls lightweight for our
classification/ranking prompts.
"""

from __future__ import annotations

import os
import json
import urllib.request
from typing import Any, Dict
from config.settings import DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL


def _ping_ollama(base_url: str, model: str) -> Dict[str, Any]:
    """Ping an Ollama server for diagnostics.

    Contacts the Ollama API at ``base_url`` to retrieve version and tag
    information and checks whether the specified ``model`` is available.
    Returns a dictionary with keys ``ok`` (bool), ``version`` (dict or
    None), ``models`` (list of model names) and ``has_model`` (bool)
    along with the original ``base_url`` and ``model`` values.  On
    failure it sets ``ok=False`` and includes an ``error`` string.
    """
    out: Dict[str, Any] = {"ok": True, "base_url": base_url, "model": model}
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/version", timeout=5) as resp:
            out["version"] = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        out.update(ok=False, error=f"Failed to reach {base_url}/api/version: {exc}")
        return out
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8")).get("models", [])
            names = [m.get("name") or m.get("model") for m in tags]
            out["models"] = names
            out["has_model"] = model in names
    except Exception as exc:
        out.update(ok=False, error=f"Failed to query tags: {exc}")
    return out


try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore

_LLM: Any = None


def _get_llm() -> Any:
    base_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if ChatOllama is None:
        raise ImportError("ChatOllama is not available; install langchain-ollama.")
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(model=model, base_url=base_url, temperature=0.0, keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
    return _LLM


def run_instruction(instruction: str) -> str:
    """Execute a task using the Ollama‑powered LangChain agent.

    This convenience wrapper constructs and invokes a LangChain agent on
    demand.  It first ensures a non‑empty prompt, then verifies that the
    Ollama server specified by the ``OLLAMA_URL`` environment variable
    (defaulting to ``http://localhost:11434``) is reachable and that
    the model specified by ``OLLAMA_MODEL`` (defaulting to ``qwen3:30b``)
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
    prompt = (instruction or "").strip()
    if not prompt:
        return (
            "No instruction provided. Please supply a natural‑language task "
            "for the agent to perform."
        )
    base_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    diag = _ping_ollama(base_url, model)
    if not diag.get("ok", False):
        return f"Cannot reach Ollama at {base_url}: {diag.get('error', 'Unknown error')}"
    if not diag.get("has_model", False):
        return (
            f"Model '{model}' not found on Ollama server. Available models: "
            f"{diag.get('models')}"
        )
    # Use a single ChatOllama instance and call invoke directly.
    try:
        llm = _get_llm()
        response = llm.invoke(prompt)  # type: ignore
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"Error executing instruction: {exc}"
    # Fallback: if the answer looks like a plan (contains an action and action_input)
    # but the agent did not execute the tool, parse and run the tool manually.
    # No tool-execution plan handling is needed for our classification prompts.
    # Persist the thought process and steps to a log file under the
    # output directory.  Each invocation appends to the log with the
    # instruction and the model's full response.  This allows users to
    # inspect how the agent reasoned about the task and which tools were
    # invoked.  If the file cannot be written, we silently ignore
    # errors.
    try:
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
    return str(answer)
