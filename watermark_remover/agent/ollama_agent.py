"""Agent orchestrator for the Watermark Remover using an Ollama-backed LLM.

This module exposes a simple function that constructs a LangChain agent
powered by a local Ollama model (such as `qwen3:30b`) and a set of tools
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

import os
from typing import Optional

from langchain.agents import initialize_agent, AgentType
from langchain_ollama import ChatOllama

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
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
        Name of the Ollama model to use (default is ``"qwen3:30b"``).  Make sure
        this model has been pulled onto the Ollama server.
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
        base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Instantiate the chat model.  We explicitly set ``keep_alive`` so the
    # model stays resident in the Ollama server between successive tool
    # invocations; this avoids the overhead of unloading and reloading the
    # large Qwen3 weights on every call.
    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        keep_alive=keep_alive,
        # Setting ``verbose`` on the model influences logging of API requests.
        verbose=verbose,
    )

    # Use the @tool-decorated callables directly.  The LangChain agent will
    # inspect their signatures and docstrings to construct the tool schemas.
    tools = [scrape_music, remove_watermark, upscale_images, assemble_pdf]

    # Construct a structured chat agent that uses function-calling style
    # reasoning (ReAct).  This agent reads the tool descriptions and
    # determines when to call them based on the user's input.  Structured
    # chat helps ensure that complex arguments (like JSON structures) are
    # passed correctly to the tools.
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
    )
    return agent_executor


def main() -> None:
    """Entry point for running the Ollama-powered Watermark Remover agent.

    This function spawns a simple command-line loop that repeatedly prompts
    the user for instructions and forwards them to the agent.  Responses
    from the agent (including any tool outputs) are printed to stdout.  To
    exit the loop, type ``exit`` or ``quit`` at the prompt.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the Watermark Remover agent")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen3:30b"),
        help="Name of the Ollama model to use (default: qwen3:30b)",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=os.environ.get("OLLAMA_URL", None),
        help="Base URL of the Ollama server (default: value of OLLAMA_URL env or http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0)",
    )
    parser.add_argument(
        "--keep-alive",
        default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        help="Keep-alive duration for the model on the server (default: 30m)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the agent",
    )
    args = parser.parse_args()

    agent = get_ollama_agent(
        model_name=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        keep_alive=args.keep_alive,
        verbose=args.verbose,
    )
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
            response = agent.run(user_input)
            print(response)
        except Exception as exc:
            print(f"Error: {exc}")
    print("Goodbye!")


if __name__ == "__main__":
    main()