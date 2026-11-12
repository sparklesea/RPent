#!/usr/bin/env bash
# RoboTwin Qwen+GPT collaborative evaluation wrapper.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "$PYTHON_BIN" = "python3" ] && [ -x "$HERE/../../.runtime/venv/bin/python" ]; then
  PYTHON_BIN="$HERE/../../.runtime/venv/bin/python"
fi

exec "$PYTHON_BIN" "$HERE/run_collab_eval.py" "$@"
