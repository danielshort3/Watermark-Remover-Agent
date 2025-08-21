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

from typing import Any, Dict, Optional, Sequence

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from watermark_remover.agent.ollama_agent import get_ollama_agent

import os
import json
import time
import urllib.request
import logging

from langchain_core.messages import BaseMessage, HumanMessage
# We avoid importing RunnableConfig from langgraph.schema because that module may not
# be available in all LangGraph installations.  Instead, we accept any mapping
# type for the config argument in the input loader.

class WMState(TypedDict, total=False):
    """State type for the LLM graph.

    - ``messages`` accumulates a sequence of chat messages.  The ``add_messages``
      reducer will append new messages when nodes return ``{"messages": ...}``.
    - ``params`` holds structured arguments (e.g. input_dir, output_pdf) for
      downstream tools or nodes.  Nodes should read from and write to this
      dictionary rather than creating arbitrary top-level keys.
    - ``result`` stores the final agent output.
    - ``received_state`` logs the keys seen by the agent node for debugging.
    """

    messages: Sequence[BaseMessage]
    params: Dict[str, Any]
    result: str
    received_state: Sequence[str]


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


# Backwards-compatibility type alias.  Older code may refer to LLMState;
# point it to WMState so the graph continues to work.
class LLMState(WMState):  # type: ignore
    pass


def agent_node(state: WMState) -> WMState:
    """Run the Ollama-backed agent using the last user message as the prompt.

    This node expects a ``messages`` list on the state containing at least one
    :class:`HumanMessage`.  It extracts the most recent user message, uses it
    as the instruction for the LangChain agent, and stores the output under
    the ``result`` key.  All structured arguments for the task (e.g.
    ``input_dir`` or ``output_pdf``) should be stored in ``params`` by
    ``input_loader_node``.
    """
    logger = logging.getLogger("wmra.agent_node")
    # Log the state keys for debugging
    keys = list(state.keys())
    logger.info("agent_node: received state keys=%s", keys)
    # Copy incoming state to avoid mutating in place
    new_state: WMState = dict(state)
    new_state["received_state"] = keys  # record for debugging
    # Extract latest user message
    messages = list(state.get("messages", []))
    instruction: Optional[str] = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            instruction = msg.content
            break
    if not instruction:
        msg = (
            "No user message found. Ensure that a HumanMessage is present in the"
            " 'messages' list before agent_node runs."
        )
        logger.warning(msg)
        new_state.update({"ok": False, "result": msg})
        return new_state
    instruction = instruction.strip()
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
        response = agent.invoke({"input": instruction})
        logger.debug("agent.invoke returned: %s", response)
    except Exception as exc:
        msg = f"Agent error: {exc}"
        logger.exception(msg)
        new_state.update({"ok": False, "result": msg, "diag": diag})
        return new_state
    elapsed = time.perf_counter() - start_time
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

def input_loader_node(state: WMState, config: Optional[Dict[str, Any]] = None) -> WMState:
    """Normalize incoming inputs into chat messages and structured params.

    When using LangGraph via the Assistants API or Studio, the initial
    user-provided dictionary is often nested under the special ``__start__``
    key on the state or under ``config.configurable.values`` in the node
    configuration.  This loader flattens those structures and converts them
    into a standard representation: natural language prompts are stored in
    ``messages`` as a list of :class:`~langchain_core.messages.HumanMessage`
    instances, while all remaining key/value pairs are stored in the
    ``params`` dictionary.  If a ``messages`` list already exists on the
    state, this loader returns an empty update to avoid overwriting it.

    Parameters
    ----------
    state : WMState
        Current graph state.  May contain a ``__start__`` mapping with user
        inputs or previously assembled ``messages``/``params``.
    config : dict, optional
        The runnable configuration provided by LangGraph.  Values under
        ``config['configurable']['values']`` (if present) are merged with
        the ``__start__`` payload to extract additional parameters when
        present.

    Returns
    -------
    WMState
        A partial state update containing new ``messages`` and/or ``params``
        keys.  If no new inputs are found, an empty dict is returned.
    """
    # Always start by preserving any existing chat messages or params.  In
    # LangGraph 0.3.x the default state reducer replaces the entire state
    # with the dict returned by this node.  If we returned an empty dict
    # here, ``messages`` and ``params`` would be lost.  Instead, copy them
    # into the update so that downstream nodes continue to see them.
    update: WMState = {}
    if state.get("messages"):
        # Preserve existing messages
        update["messages"] = list(state.get("messages", []))
    if state.get("params"):
        # Preserve existing structured parameters
        update["params"] = dict(state.get("params", {}))  # type: ignore[dict-assign]
    # Gather potential inputs from three sources:
    # 1) The reserved ``__start__`` key when using the Assistants API.
    # 2) The ``values`` key used by the LangGraph SDK when calling runs directly.
    # 3) The ``config['configurable']['values']`` mapping provided via runnable config.
    start_payload: Dict[str, Any] = {}
    start_raw = state.get("__start__")
    if isinstance(start_raw, dict):
        start_payload = dict(start_raw)
    values_payload: Dict[str, Any] = {}
    values_raw = state.get("values")
    if isinstance(values_raw, dict):
        values_payload = dict(values_raw)
    cfg_values: Dict[str, Any] = {}
    try:
        if config and isinstance(config, dict):  # type: ignore[redundant-expr]
            # The configuration is treated as a generic mapping; get nested values defensively
            cfg = config.get("configurable", {})  # type: ignore[index]
            if isinstance(cfg, dict):
                vals = cfg.get("values", {})  # type: ignore[index]
                if isinstance(vals, dict):
                    cfg_values = dict(vals)
    except Exception:
        # Ignore malformed config
        cfg_values = {}
    # Combine the two dicts, giving precedence to __start__ over config values
    merged: Dict[str, Any] = {**cfg_values, **start_payload, **values_payload}
    if not merged:
        # Nothing else to merge; return the preserved update (messages/params)
        return update
    # Extract the natural-language prompt.  Accept both 'instruction' and
    # 'prompt' for backwards compatibility.
    instruction = merged.pop("instruction", None) or merged.pop("prompt", None)
    if instruction:
        # Override any existing messages with the new instruction
        update["messages"] = [HumanMessage(content=str(instruction))]
    # The remainder of the merged dict are structured parameters
    if merged:
        # Start with any preserved params from the update
        current_params: Dict[str, Any] = dict(update.get("params", {}))  # type: ignore[dict-assign]
        # Only add keys that aren't already present in params
        for k, v in merged.items():
            current_params.setdefault(k, v)
        update["params"] = current_params
    return update


def compile_graph() -> Any:
    """Construct and compile the LangGraph for the Ollama agent.

    This graph contains two nodes: an ``input_loader`` that normalises
    incoming API payloads into chat messages and structured parameters,
    and an ``agent`` node that delegates execution to the Ollama-backed
    chat agent.  The state type is :class:`WMState`, which defines
    keys for accumulated ``messages``, ``params`` and the ``result``.

    Returns
    -------
    compiled_graph
        A compiled graph ready for execution or serving via LangGraph.
    """
    # Use a plain dict for the state type to allow arbitrary top-level keys
    # such as 'values' or '__start__'.  Using a TypedDict here would
    # discard unknown keys before they reach the input loader.
    graph = StateGraph(dict)  # type: ignore[type-arg]
    # Register nodes
    graph.add_node("input_loader", input_loader_node)
    graph.add_node("agent", agent_node)
    # Define control flow: START -> input_loader -> agent -> END
    graph.add_edge(START, "input_loader")
    graph.add_edge("input_loader", "agent")
    graph.add_edge("agent", END)
    return graph.compile()


# Export a compiled graph instance as ``graph`` for LangGraph CLI discovery
graph = compile_graph()