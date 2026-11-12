#!/usr/bin/env bash
# ECC RoboTwin stage 1 wrapper: Qwen-only exploration on two GPUs by default.
# Usage:
#   scripts/ecc_robotwin/run_ecc_stage1.sh RUN_DIR MEMORY_DIR \
#     --assets-path ASSETS --model-path MODEL
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN_DIR="${1:?run dir}"
MEMORY_DIR="${2:?memory dir}"
shift 2

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "$PYTHON_BIN" = "python3" ] && [ -x "$ROOT/.runtime/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT/.runtime/venv/bin/python"
fi

exec "$PYTHON_BIN" "$HERE/run_ecc_stage1.py" "$RUN_DIR" "$MEMORY_DIR" "$@"
