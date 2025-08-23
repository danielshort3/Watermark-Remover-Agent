"""Helpers for invoking the Ollama‑powered chat agent.

This module exposes a utility to run natural‑language instructions
through a LangChain agent backed by an Ollama model and capture the
LLM's reasoning.  All reasoning and final responses are written to
the current run's log directory (see ``WMRA_LOG_DIR``) to aid in
debugging and transparency.
"""

from __future__ import annotations

import os
import json
import urllib.request
from typing import Any, Dict

from watermark_remover.agent.ollama_agent import get_ollama_agent

# -----------------------------------------------------------------------------

def _ping_ollama(base_url: str, model: str) -> Dict[str, Any]:
    """Check availability of the Ollama server and model."""
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
            out["models"] = tags
            out["has_model"] = model in tags
    except Exception as exc:
        out.update(ok=False, error=f"Failed to fetch models: {exc}")
    return out


def run_instruction(prompt: str, model: str = "qwen3:30b", base_url: str | None = None) -> str:
    """Execute a user instruction via the Ollama agent with reasoning logging.

    When ``verbose=True`` the agent prints its reasoning steps to stdout.
    We capture that output and append it to a ``thoughts_and_steps.log`` file
    under the log directory.  The function returns the agent's final output
    string.
    """
    # Determine log directory
    log_dir = os.environ.get("WMRA_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(log_dir, exist_ok=True)
    # Try to build the agent
    try:
        # We enable verbose output so the agent prints reasoning to stdout.
        agent = get_ollama_agent(model_name=model, base_url=base_url, verbose=True)
    except Exception as exc:
        return f"Failed to create Ollama agent: {exc}"
    # Execute the instruction end-to-end using agent.run().  This will
    # invoke tools automatically rather than just returning a plan.
    try:
        result = agent.run(prompt)
    except Exception as exc:
        return f"Error executing instruction: {exc}"
    # Extract answer; intermediate steps are printed to stdout when verbose=True.
    answer = str(result)
    steps = []  # We no longer capture intermediate steps directly
    # Log reasoning and final answer
    try:
        log_path = os.path.join(log_dir, "thoughts_and_steps.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Instruction: {prompt}\n")
            if steps:
                for idx, step in enumerate(steps, 1):
                    try:
                        thought = step.get("thought")
                        action = step.get("action")
                        action_input = step.get("action_input")
                        observation = step.get("observation")
                    except Exception:
                        thought = action = action_input = observation = None
                    f.write(f"Step {idx}:\n")
                    if thought:
                        f.write(f"  Thought: {thought}\n")
                    if action:
                        f.write(f"  Action: {action}\n")
                    if action_input:
                        f.write(f"  Action Input: {action_input}\n")
                    if observation:
                        f.write(f"  Observation: {observation}\n")
            else:
                # If intermediate steps are unavailable, log the raw result for transparency
                try:
                    import json
                    f.write("No intermediate steps provided. Raw result:\n")
                    f.write(json.dumps(result, indent=2) + "\n")
                except Exception:
                    f.write("No intermediate steps provided. Result could not be serialized.\n")
            f.write(f"Answer: {answer}\n\n")
    except Exception:
        pass
    return str(answer)