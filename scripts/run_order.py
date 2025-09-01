#!/usr/bin/env python3
"""Run the order-of-worship graph from the CLI.

Usage:
  python scripts/run_order.py --instruction "Process the 08_31_2025 service PDF"

Options:
  --instruction, -i   Instruction string for the parser (required)
  --max-procs, -p     Concurrency cap (default: all songs)
  --top-n, -n         Max candidates per song (default: 3)
  --sequential        Force sequential mode (ORDER_PARALLEL=0)
  --debug             Enable verbose debug artifacts
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from watermark_remover.agent.order_of_worship_graph import graph  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Run order-of-worship graph")
    ap.add_argument("--instruction", "-i", required=True)
    ap.add_argument("--max-procs", "-p", type=int, default=0)
    ap.add_argument("--top-n", "-n", type=int, default=3)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.sequential:
        os.environ["ORDER_PARALLEL"] = "0"
    else:
        os.environ.setdefault("ORDER_PARALLEL", "1")
    if args.max_procs:
        os.environ["ORDER_MAX_PROCS"] = str(args.max_procs)

    state = {
        "user_input": args.instruction,
        "top_n": int(args.top_n),
        "debug": bool(args.debug),
    }
    out = graph.invoke(state)
    # Print a concise summary
    final_pdfs = out.get("final_pdfs") if isinstance(out, dict) else None
    if isinstance(final_pdfs, list):
        print(f"Generated {sum(1 for p in final_pdfs if p)} PDFs")
    else:
        print("Run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

