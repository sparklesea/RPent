#!/usr/bin/env bash
# RoboTwin Best-of-M + GPT verifier evaluation wrapper.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "$PYTHON_BIN" = "python3" ] && [ -x "$HERE/../../.runtime/venv/bin/python" ]; then
  PYTHON_BIN="$HERE/../../.runtime/venv/bin/python"
fi

exec "$PYTHON_BIN" "$HERE/run_candidate_verify_eval.py" "$@"
