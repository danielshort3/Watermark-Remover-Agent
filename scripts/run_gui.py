#!/usr/bin/env python3
"""Launch the Order of Worship GUI.

Usage:
  python scripts/run_gui.py --host 0.0.0.0 --port 7860
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

from watermark_remover.gui.order_gui import launch  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Order of Worship GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument("--share", dest="share", action="store_true")
    share_group.add_argument("--no-share", dest="share", action="store_false")
    parser.set_defaults(share=True)
    args = parser.parse_args()
    launch(host=args.host, port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
