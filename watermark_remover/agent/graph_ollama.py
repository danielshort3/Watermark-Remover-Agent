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

    The agent node performs several checks before invoking the LLM:

    * Validates that an ``instruction`` is present somewhere in the input
      state.  It recursively searches nested mappings for an ``instruction``
      key to accommodate various state shapes produced by LangGraph or API
      wrappers.  If none is found, it returns a helpful error.
    * Pings the Ollama server specified by the environment variables (or
      defaults) to verify connectivity and model availability.  If the
      server cannot be reached or the model isn't available, it returns
      diagnostics describing the failure.
    * If connectivity checks pass, it initialises the agent via
      :func:`get_ollama_agent` and runs it on the instruction.  Any
      exceptions raised during agent construction or invocation are
      captured and returned as part of the response.

    The final result (success or error message) is stored under the
    ``result`` key.  Additional fields ``ok``, ``diag`` and ``elapsed_s`` may
    be present to aid debugging.
    """
    logger = logging.getLogger("wmra.agent_node")

    def _extract_from_state(container: Any, key: str) -> Optional[str]:
        """Recursively search for ``key`` in nested mappings and lists.

        Returns the first found value or ``None`` if not present.
        """
        if isinstance(container, dict):
            if key in container:
                return container[key]
            for v in container.values():
                found = _extract_from_state(v, key)
                if found is not None:
                    return found
        elif isinstance(container, (list, tuple, set)):
            for item in container:
                found = _extract_from_state(item, key)
                if found is not None:
                    return found
        return None

    # Log the incoming state at debug level.  Fallback to repr if JSON fails.
    try:
        serialized_state = json.dumps(state, default=str)
        logger.debug("agent_node received state (json): %s", serialized_state)
    except Exception:
        logger.debug("agent_node received state (repr): %r", state)
    # Always log the received keys at info level so they appear even when DEBUG is off
    try:
        keys = list(state.keys())
    except Exception:
        keys = []
    logger.info("agent_node: received state keys=%s", keys)
    # Copy incoming state to avoid mutating in place
    new_state: LLMState = dict(state)
    new_state["received_state"] = keys
    # Recursively extract the instruction
    raw_instruction = _extract_from_state(state, "instruction")
    instruction = (raw_instruction or "").strip() if isinstance(raw_instruction, str) else ""
    if not instruction:
        msg = (
            "No 'instruction' provided. Please supply an 'instruction' string in the"
            " input JSON (e.g. {\"instruction\": \"Say ONLY READY.\"}). "
            f"Received keys: {keys}"
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


# -----------------------------------------------------------------------------
# Graph definition
#
# The agent graph expects the user's input to be supplied under the ``instruction``
# key at the top level of the state.  However, when using the LangGraph API, the
# initial state is often nested under a reserved ``__start__`` key.  The
# ``input_loader_node`` defined below merges any ``__start__`` values into the
# top‑level state before execution continues to the agent node.  This ensures
# that fields like ``instruction``, ``input_dir`` or ``output_pdf`` are
# accessible regardless of how the API structures the incoming state.

def input_loader_node(state: LLMState) -> LLMState:
    """Merge nested ``__start__`` values into the top-level state.

    The LangGraph API wraps the user-provided input dict under a ``__start__``
    key.  This node detects such a mapping and flattens it into the main state
    so that subsequent nodes can access the instruction and other parameters
    directly.
    """
    new_state: LLMState = dict(state)
    start_payload = state.get("__start__")
    if isinstance(start_payload, dict):
        # Merge values without overwriting existing top-level keys
        for k, v in start_payload.items():
            new_state.setdefault(k, v)
    return new_state


def compile_graph() -> Any:
    """Construct and compile the LangGraph for the Ollama agent.

    Returns
    -------
    compiled_graph
        A compiled graph that can be executed or served via the LangGraph API.
    """
    graph = StateGraph(LLMState)
    # First flatten any nested input payload under '__start__'
    graph.add_node("input_loader", input_loader_node)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "input_loader")
    graph.add_edge("input_loader", "agent")
    graph.add_edge("agent", END)
    return graph.compile()


# Export a compiled graph instance as ``graph`` for LangGraph CLI discovery
graph = compile_graph()