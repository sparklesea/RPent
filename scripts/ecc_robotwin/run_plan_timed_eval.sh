#!/usr/bin/env bash
# Launch the RoboTwin GPT-plan / Qwen-execute timed-replan baseline.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${RPENT_PYTHON:-$ROOT/.runtime/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "python interpreter not found; set RPENT_PYTHON" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$HERE/run_plan_timed_eval.py" "$@"
