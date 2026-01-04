#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR=""
if [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
elif [[ -d "venv" ]]; then
  VENV_DIR="venv"
fi

if [[ -n "$VENV_DIR" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
else
  echo "WARN: virtualenv not found (expected .venv or venv)." >&2
fi

HOST="${WMRA_GUI_HOST:-0.0.0.0}"
PORT="${WMRA_GUI_PORT:-7860}"
SHARE="${WMRA_GUI_SHARE:-true}"

if [[ "${SHARE}" == "false" || "${SHARE}" == "0" ]]; then
  python scripts/run_gui.py --host "$HOST" --port "$PORT" --no-share
else
  python scripts/run_gui.py --host "$HOST" --port "$PORT"
fi
