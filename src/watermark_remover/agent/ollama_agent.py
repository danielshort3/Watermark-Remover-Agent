"""Agent orchestrator for the Watermark Remover using an Ollama‑backed LLM.

This module exposes a simple function that constructs a LangChain agent
powered by a local Ollama model (such as ``qwen3:30b``) and a set of tools
defined in :mod:`watermark_remover.agent.tools`.  The agent can reason
about user instructions in natural language and decide when to call
individual tools like ``scrape_music``, ``remove_watermark``, ``upscale_images``
or ``assemble_pdf``.  For example, a user might ask the agent to
download a particular piece of sheet music, clean up the watermark,
upscale it and assemble the result into a PDF.  The agent will use
the LLM's reasoning capabilities to invoke the appropriate tools in
sequence without the user having to know the implementation details.

To use this agent, you need to have an Ollama server running and have
downloaded the desired model (by default ``qwen3:30b``) via the
``ollama pull`` command.  The agent will connect to the server at
``http://localhost:11434`` by default; this can be customised via the
``OLLAMA_URL`` environment variable.  See the project README for
instructions on installing and running Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Any, Optional
from config.settings import DEFAULT_OLLAMA_URL

# AgentType and initialize_agent are imported lazily within get_ollama_agent to
# avoid raising ImportError on module import when langchain is missing.  See
# get_ollama_agent below.

# ChatOllama is part of the optional langchain-ollama package.  Try to import
# it, but allow this module to be imported even if the dependency is
# unavailable.  When ChatOllama cannot be imported the agent will
# subsequently raise an ImportError when invoked.
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
    ensure_order_pdf,
)


def get_ollama_agent(
    model_name: str = "qwen3:30b",
    *,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    keep_alive: str = "30m",
    verbose: bool = False,
) -> object:
    """Return a LangChain agent backed by an Ollama chat model.

    Parameters
    ----------
    model_name : str, optional
        Name of the Ollama model to use (default is ``"qwen3:30b"``).  Make
        sure this model has been pulled onto the Ollama server.
    base_url : str, optional
        Base URL of the Ollama server.  If not provided, the environment
        variable ``OLLAMA_URL`` is consulted, falling back to
        ``"http://localhost:11434"`` if unset.
    temperature : float, optional
        Sampling temperature for the model (default 0.0 for deterministic
        outputs).
    keep_alive : str, optional
        How long the model will stay loaded in the Ollama server between
        requests.  See Ollama documentation for valid values (default
        ``"30m"``).
    verbose : bool, optional
        Whether the agent should print its reasoning steps to stdout.

    Returns
    -------
    AgentExecutor
        A runnable LangChain agent configured to interact with the given
        Ollama model and the Watermark Remover tools.
    """
    # Determine the base URL for the Ollama server.  Users can override this
    # via the function argument or the OLLAMA_URL environment variable.
    if base_url is None:
        base_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    # Ensure the required dependencies are available.  We lazily import
    # AgentType and initialize_agent to allow this module to be imported
    # without langchain installed.  The chat model is also optional.
    if ChatOllama is None:
        raise ImportError(
            "The langchain-ollama package is not installed. "
            "Install it to enable Ollama-backed agents."
        )
    try:
        from langchain.agents import AgentType, initialize_agent  # type: ignore[import]
        _legacy_agent_available = True
    except Exception:
        AgentType = None  # type: ignore[assignment]
        initialize_agent = None  # type: ignore[assignment]
        _legacy_agent_available = False

    # Instantiate the chat model.  We explicitly set ``keep_alive`` so the
    # model stays resident in the Ollama server between successive tool
    # invocations; this avoids the overhead of unloading and reloading the
    # large Qwen3 weights on every call.
    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        keep_alive=keep_alive,
        verbose=verbose,
    )
    # Use the @tool‑decorated callables directly.  The LangChain agent will
    # inspect their signatures and docstrings to construct the tool schemas.
    tools = [scrape_music, remove_watermark, upscale_images, assemble_pdf, ensure_order_pdf]

    if _legacy_agent_available and initialize_agent is not None and AgentType is not None:
        # Construct a structured chat agent that uses the legacy initialize_agent API.
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        )

    # Fall back to the modern tool-calling agent available in LangChain >= 0.2.
    try:
        from langchain.agents import create_agent  # type: ignore[import]
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
        )  # type: ignore[import]
    except Exception as exc:
        raise ImportError(
            "LangChain agent interfaces are unavailable. Install both "
            "langchain and langchain-community to enable agent construction."
        ) from exc

    system_msg = (
        "You are the Watermark Remover agent. Decide which tools to call based on "
        "the user's request. Always call ensure_order_pdf before processing an "
        "order of worship PDF so the source file is copied into output/orders."
    )
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_msg,
        debug=verbose,
    )

    class _GraphAgentWrapper:
        """Adapter that exposes an invoke() signature similar to AgentExecutor."""

        def __init__(self, compiled_graph):
            self._graph = compiled_graph

        def invoke(self, inputs: object) -> str:
            prompt: str | None = None
            messages: list[BaseMessage] | None = None
            if isinstance(inputs, dict):
                prompt = inputs.get("input") or inputs.get("prompt")  # type: ignore[arg-type]
                raw_messages = inputs.get("messages")
                if isinstance(raw_messages, list):
                    tmp: list[BaseMessage] = []
                    for item in raw_messages:
                        if isinstance(item, BaseMessage):
                            tmp.append(item)
                        elif isinstance(item, dict):
                            role = str(item.get("role") or "user").lower()
                            content = item.get("content", "")
                            if role in {"user", "human"}:
                                tmp.append(HumanMessage(content=str(content)))
                            else:
                                tmp.append(AIMessage(content=str(content)))
                        else:
                            tmp.append(HumanMessage(content=str(item)))
                    messages = tmp
            elif isinstance(inputs, str):
                prompt = inputs
            if messages is None:
                if not prompt:
                    raise ValueError("Expected an 'input' string when invoking the agent.")
                messages = [HumanMessage(content=str(prompt))]
            if not messages:
                raise ValueError("Expected an 'input' string when invoking the agent.")
            result = self._graph.invoke({"messages": messages})
            messages = result.get("messages") if isinstance(result, dict) else None
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, AIMessage):
                        return message.content
                if messages:
                    return str(messages[-1])
                return ""
            return str(result)

    return _GraphAgentWrapper(graph)


def _ping(base_url: str, path: str) -> dict:
    """Helper to fetch JSON from an Ollama API endpoint."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def diag() -> None:
    """Diagnostic entry point to verify Ollama connectivity and model availability."""
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:30b")
    out: dict[str, Any] = {"base_url": base, "model": model}
    try:
        out["version"] = _ping(base, "/api/version")
        tags = _ping(base, "/api/tags").get("models", [])
        names = [m.get("name") or m.get("model") for m in tags]
        out["models"] = names
        out["has_model"] = model in names
        # Perform a simple generation to confirm model works
        llm = ChatOllama(model=model, base_url=base)
        out["roundtrip"] = llm.invoke("Say READY.").content
    except Exception as exc:
        out["error"] = repr(exc)
    print(json.dumps(out, indent=2))


def repl(agent) -> None:
    """Run an interactive loop for the agent, reading user instructions from stdin."""
    print(
        "\nWatermark Remover Agent (Ollama-powered)\n"
        "Enter a command describing what you want to do, or 'exit' to quit.\n"
    )
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        try:
            # Prefer invoke to run for improved API compatibility
            response = agent.invoke({"input": user_input})
            # Extract text if response is structured
            if isinstance(response, dict):
                result_text = response.get("output") or response.get("result") or response
            else:
                result_text = response
            print(result_text)
        except Exception as exc:
            print(f"Error: {exc}")
    print("Goodbye!")


def run_once(agent, instruction: str) -> None:
    """Run a single instruction with the agent and print the result."""
    try:
        resp = agent.invoke({"input": instruction})
        print(resp)
    except Exception as exc:
        print(f"Error: {exc}")
