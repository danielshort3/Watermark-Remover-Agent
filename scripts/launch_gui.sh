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

kill_port_process() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -lptn "sport = :$port" 2>/dev/null | awk -F'pid=|,' 'NR>1 {print $2}' | tr -d ' ' | sort -u || true)"
  elif command -v netstat >/dev/null 2>&1; then
    pids="$(netstat -lptn 2>/dev/null | awk -v port=":$port" '$4 ~ port {gsub("/.*","",$7); print $7}' | sort -u || true)"
  fi

  if [[ -n "$pids" ]]; then
    echo "Killing existing process(es) on port $port: $pids"
    kill $pids 2>/dev/null || true
    sleep 0.5
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi
}

kill_port_process "$PORT"

if [[ "${SHARE}" == "false" || "${SHARE}" == "0" ]]; then
  python scripts/run_gui.py --host "$HOST" --port "$PORT" --no-share
else
  python scripts/run_gui.py --host "$HOST" --port "$PORT"
fi
