"""
LangGraph definition that wraps the Ollama-powered chat agent.

This graph exposes a single node that accepts a natural-language
instruction from the user (under the ``instruction`` key of the
graph's state) and delegates execution to an LLM-based agent backed
by a local Ollama server.  The agent can leverage all of the tools
defined in :mod:`watermark_remover.agent.tools` (e.g. scraping,
watermark removal, upscaling, assembling) to satisfy the user's
request.  The result of the agent run (the agent's final answer) is
stored under the ``result`` key in the state.

Users can invoke this graph via LangGraph Studio or the API by
providing a JSON object with an ``instruction`` string.  For example:

    {
        "instruction": "Download sheet music for Fur Elise, remove the 
        watermark, upscale it and save it as a PDF"
    }

The graph will then spin up the Ollama agent (if not already loaded)
and produce a human-readable response describing the outcome of the
requested operations.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from watermark_remover.agent.ollama_agent import get_ollama_agent

import os
import json
import time
import urllib.request
import logging


def _ping_ollama(base_url: str, model: str) -> Dict[str, Any]:
    """Ping an Ollama server for basic diagnostics.

    This helper tries to reach the Ollama API at the given ``base_url``
    and returns a dictionary describing the server version and which
    models are available.  If any call fails, it sets ``ok=False`` and
    records the error message.  The returned dict always contains
    ``base_url`` and ``model`` as keys.

    Parameters
    ----------
    base_url : str
        Base URL of the Ollama server (e.g. ``http://localhost:11434``).
    model : str
        Name of the model to check for availability (e.g. ``"qwen3:30b"``).

    Returns
    -------
    dict
        A dictionary with keys ``ok`` (bool), ``version`` (dict or None),
        ``models`` (list of model names) and ``has_model`` (bool) when
        available.  If ``ok`` is False, an ``error`` string will be
        present describing the failure.
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


class LLMState(Dict[str, Any]):
    """State passed between nodes in the LLM agent graph.

    Keys:

    - ``instruction``: (str) the natural-language instruction provided by 
      the user.
    - ``result``: (str) the textual output produced by the LLM agent.  This 
      will only be populated after the agent node has run.
    """


def agent_node(state: LLMState) -> LLMState:
    """Run the Ollama-backed agent on the user's instruction with diagnostics.

    This implementation performs several checks before invoking the LLM:

    * It validates that an ``instruction`` is present in the state.  If
      missing or empty, a helpful message is returned immediately.
    * It pings the Ollama server specified by the environment variables
      (or defaults) to verify connectivity and model availability.  If the
      server cannot be reached or the model isn't available, a
      descriptive error and diagnostics are returned.
    * When the checks pass, it initialises the agent via
      :func:`get_ollama_agent` and runs it on the instruction.  Any
      exceptions raised during agent construction or invocation are
      captured and returned as part of the response.

    The final result (success or error message) is stored under the
    ``result`` key.  Additional fields ``ok``, ``diag`` and
    ``elapsed_s`` may be present to aid debugging.

    Parameters
    ----------
    state : LLMState
        The current state of the graph.  Must contain an ``instruction`` key
        with the user's request.

    Returns
    -------
    LLMState
        A new state dict containing the ``result`` of the agent run and
        diagnostics about the call.
    """
    logger = logging.getLogger("wmra.agent_node")
    # Log the entire incoming state at debug level for thorough diagnostics
    try:
        serialized_state = json.dumps(state, default=str)
        logger.debug("agent_node received state: %s", serialized_state)
    except Exception:
        # Fallback if state contains non-serialisable objects
        logger.debug("agent_node received state keys: %s", list(state.keys()))
    # Always log the received keys at info level so they appear even when DEBUG is off
    logger.info("agent_node: received state keys=%s", list(state.keys()))
    # Copy incoming state to avoid mutating in place
    new_state: LLMState = dict(state)
    # Always record which keys were received for debugging
    new_state["received_state"] = list(state.keys())
    # Extract the instruction.  Look on the top-level first, then search
    # nested mappings for a key called "instruction" to handle state wrappers.
    instruction_raw = state.get("instruction")
    if instruction_raw is None:
        # Sometimes the state is nested (e.g. under '__start__' or other keys).
        for val in state.values():
            if isinstance(val, dict) and "instruction" in val:
                instruction_raw = val.get("instruction")
                break
    instruction = (instruction_raw or "").strip()
    if not instruction:
        msg = (
            "No 'instruction' provided. Please supply an 'instruction' string in the"
            " input JSON (e.g. {\"instruction\": \"Say ONLY READY.\"}). "
            f"Received keys: {list(state.keys())}"
        )
        logger.warning(msg)
        new_state.update({"ok": False, "result": msg})
        return new_state

    # Determine server URL and model from environment, with sensible defaults
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:30b")

    # Ping the Ollama server and check for the model
    diag = _ping_ollama(base_url, model)
    if not diag.get("ok", False):
        msg = f"Cannot reach Ollama at {base_url}: {diag.get('error', 'Unknown error')}"
        logger.error(msg)
        new_state.update({"ok": False, "result": msg, "diag": diag})
        return new_state
    if not diag.get("has_model", False):
        msg = f"Model '{model}' not found on Ollama server. Available models: {diag.get('models')}"
        logger.error(msg)
        new_state.update({"ok": False, "result": msg, "diag": diag})
        return new_state

    # Instantiate and run the agent
    start_time = time.perf_counter()
    try:
        agent = get_ollama_agent(model_name=model, base_url=base_url)
    except Exception as exc:
        msg = f"Failed to initialise agent: {exc}"
        logger.exception(msg)
        new_state.update({"ok": False, "result": msg, "diag": diag})
        return new_state
    try:
        # Use .invoke to call the agent.  Some agents return a dict with
        # 'output' or 'result', others return a plain string.  We'll log the
        # raw response for debugging.
        response = agent.invoke({"input": instruction})
        logger.debug("agent.invoke returned: %s", response)
    except Exception as exc:
        msg = f"Agent error: {exc}"
        logger.exception(msg)
        new_state.update({"ok": False, "result": msg, "diag": diag})
        return new_state
    elapsed = time.perf_counter() - start_time
    # Ensure response is string-like (some agents return dict or messages)
    if isinstance(response, dict):
        result_text = str(response.get("output") or response.get("result") or response)
    else:
        result_text = str(response)
    new_state.update({"ok": True, "result": result_text, "diag": diag, "elapsed_s": round(elapsed, 3)})
    return new_state


def compile_graph() -> Any:
    """Construct and compile the LangGraph for the Ollama agent.

    Returns
    -------
    compiled_graph
        A compiled graph that can be executed or served via the LangGraph API.
    """
    graph = StateGraph(LLMState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


# Export a compiled graph instance as ``graph`` for LangGraph CLI discovery
graph = compile_graph()