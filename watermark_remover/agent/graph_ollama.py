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

.. code-block:: json

    {
        "instruction": "Download sheet music for Fur Elise, remove the watermark, upscale it and save it as a PDF"
    }

The graph will then spin up the Ollama agent (if not already loaded)
and produce a human-readable response describing the outcome of the
requested operations.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from watermark_remover.agent.ollama_agent import get_ollama_agent


class LLMState(Dict[str, Any]):
    """State passed between nodes in the LLM agent graph.

    Keys:

    - ``instruction``: (str) the natural-language instruction provided by the user.
    - ``result``: (str) the textual output produced by the LLM agent.  This will
      only be populated after the agent node has run.
    """


def agent_node(state: LLMState) -> LLMState:
    """Run the Ollama-backed agent on the user's instruction.

    This node uses :func:`get_ollama_agent` to instantiate (or retrieve) an
    LLM-driven agent and then calls it with the instruction stored in
    ``state['instruction']``.  The agent is configured to call any of the
    available tools (scrape_music, remove_watermark, etc.) as needed to
    accomplish the request.  The final response from the agent is saved
    under the ``result`` key in the returned state.

    Parameters
    ----------
    state : LLMState
        The current state of the graph.  Must contain an ``instruction`` key
        with the user's request.

    Returns
    -------
    LLMState
        A new state dict containing the ``result`` of the agent run.
    """
    # Copy incoming state to avoid mutating in place
    new_state: LLMState = dict(state)
    instruction = state.get("instruction")
    if not instruction:
        # Nothing to do; return state unchanged
        new_state["result"] = ""
        return new_state

    # Create the agent.  This will connect to the Ollama server specified
    # by environment variables (OLLAMA_URL, OLLAMA_MODEL, etc.).  If the
    # model hasn't been pulled yet, Ollama will raise an error which
    # bubbles up here.
    agent = get_ollama_agent()
    try:
        response = agent.run(instruction)
    except Exception as exc:
        response = f"Agent error: {exc}"
    new_state["result"] = response
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