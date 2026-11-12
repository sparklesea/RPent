#!/usr/bin/env bash
# Start one shared LingBot VLA server on a selected GPU.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

GPU="${1:?GPU is required}"
PORT="${2:?port is required}"
MODEL_PATH="${3:?model path is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [ -z "$PYTHON_BIN" ] && [ -x "$ROOT/.runtime/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT/.runtime/venv/bin/python"
fi
: "${PYTHON_BIN:?python entrypoint not found; set PYTHON_BIN or activate an environment}"

MODEL_PATH="$(realpath "$MODEL_PATH")"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN25_PATH="$MODEL_PATH/qwen_base"

exec "$PYTHON_BIN" "$ROOT/robots/robotwin/vla_server.py" \
  --model-path "$MODEL_PATH" \
  --use-length 50 \
  --port "$PORT" \
  --norm-path "$MODEL_PATH/norm_stats/robotwin_eef.json" \
  --lingbot-robot-config "$MODEL_PATH/configs/robot_configs/robotwin_eef.yaml" \
  --parent-watch
