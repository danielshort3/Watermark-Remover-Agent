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
from typing import Optional, Any

from langchain.agents import initialize_agent, AgentType
from langchain_ollama import ChatOllama

from watermark_remover.agent.tools import (
    scrape_music,
    remove_watermark,
    upscale_images,
    assemble_pdf,
)

import json
import urllib.request
import argparse


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

def _ping(base_url: str, path: str) -> dict:
    """Helper to fetch JSON from an Ollama API endpoint.

    Parameters
    ----------
    base_url : str
        The root URL of the Ollama server.
    path : str
        API path starting with '/api'.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def diag() -> None:
    """Diagnostic entry point to verify Ollama connectivity and model availability.

    This function queries the Ollama server for its version, lists available
    models and checks whether the requested model is present.  It also
    performs a simple round-trip call to the model to ensure it can generate
    a response.  The diagnostic information is printed as formatted JSON.
    """
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
    """Run an interactive loop for the agent, reading user instructions from stdin.

    Parameters
    ----------
    agent : AgentExecutor
        The agent to invoke for each user input.
    """
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
    """Run a single instruction with the agent and print the result.

    Parameters
    ----------
    agent : AgentExecutor
        The agent to use.
    instruction : str
        The natural-language instruction to process.
    """
    try:
        resp = agent.invoke({"input": instruction})
    except Exception as exc:
        print(f"Error: {exc}")
        return
    if isinstance(resp, dict):
        result_text = resp.get("output") or resp.get("result") or resp
    else:
        result_text = resp
    print(result_text)


def main() -> None:
    """Entry point for running the Watermark Remover agent.

    This function uses subcommands to expose different modes:

    * ``diag``: print diagnostics about the Ollama server and model.
    * ``repl``: run an interactive prompt powered by the agent.
    * ``run``: execute a single instruction passed via ``--instruction``.

    If no subcommand is provided, ``repl`` is the default.
    """
    parser = argparse.ArgumentParser(description="Watermark Remover Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)
    # diag subcommand
    subparsers.add_parser("diag", help="Check connectivity to Ollama and model availability")
    # repl subcommand
    repl_parser = subparsers.add_parser("repl", help="Run the interactive agent REPL")
    repl_parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen3:30b"),
        help="Name of the Ollama model to use (default: qwen3:30b)",
    )
    repl_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=os.environ.get("OLLAMA_URL", None),
        help="Base URL of the Ollama server (default: value of OLLAMA_URL env or http://localhost:11434)",
    )
    repl_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0)",
    )
    repl_parser.add_argument(
        "--keep-alive",
        default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        help="Keep-alive duration for the model on the server (default: 30m)",
    )
    repl_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the agent",
    )
    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run the agent on a single instruction")
    run_parser.add_argument(
        "instruction",
        help="Natural-language instruction for the agent to execute",
    )
    run_parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen3:30b"),
        help="Name of the Ollama model to use (default: qwen3:30b)",
    )
    run_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=os.environ.get("OLLAMA_URL", None),
        help="Base URL of the Ollama server (default: value of OLLAMA_URL env or http://localhost:11434)",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0)",
    )
    run_parser.add_argument(
        "--keep-alive",
        default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        help="Keep-alive duration for the model on the server (default: 30m)",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the agent",
    )
    args = parser.parse_args()

    # If no command is provided, default to repl
    cmd = args.command or "repl"
    if cmd == "diag":
        diag()
        return
    elif cmd in {"repl", "run"}:
        # Determine base_url and model from command-specific args (may be None)
        base_url = getattr(args, "ollama_url", None)
        model_name = getattr(args, "model", os.environ.get("OLLAMA_MODEL", "qwen3:30b"))
        temp = getattr(args, "temperature", 0.0)
        keep_alive = getattr(args, "keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        verbose = getattr(args, "verbose", False)
        agent = get_ollama_agent(
            model_name=model_name,
            base_url=base_url,
            temperature=temp,
            keep_alive=keep_alive,
            verbose=verbose,
        )
        if cmd == "repl":
            repl(agent)
            return
        else:  # cmd == "run"
            run_once(agent, args.instruction)
            return
    else:
        parser.error(f"Unknown command: {cmd}")



if __name__ == "__main__":
    main()