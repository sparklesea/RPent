#!/usr/bin/env bash
# Launch the RoboTwin BRACE-style event-triggered replanning baseline.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${RPENT_PYTHON:-$ROOT/.runtime/venv/bin/python}"

exec "$PYTHON_BIN" "$HERE/run_brace_eval.py" "$@"
