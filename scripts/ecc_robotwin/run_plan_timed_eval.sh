#!/usr/bin/env bash
# Launch the RoboTwin GPT-plan / Qwen-execute timed-replan baseline.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${RPENT_PYTHON:-$ROOT/.runtime/venv/bin/python}"

exec "$PYTHON_BIN" "$HERE/run_plan_timed_eval.py" "$@"
