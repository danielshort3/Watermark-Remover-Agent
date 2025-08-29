#!/usr/bin/env python3
"""Convenience script to run a single natural-language instruction via the Ollama agent.

Usage:
  python scripts/run_agent.py "Download 'Fur Elise' for French Horn in G and produce a PDF"
"""
import os
import sys

# Ensure src/ is importable when running from repo root
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from watermark_remover.agent.graph_ollama import run_instruction  # type: ignore


def main() -> int:
    if len(sys.argv) < 2:
        print("Please provide an instruction string.")
        return 2
    instruction = sys.argv[1]
    out = run_instruction(instruction)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

