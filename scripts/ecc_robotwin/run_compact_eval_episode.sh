#!/usr/bin/env bash
# One compact-tool RoboTwin Qwen+GPT evaluation episode.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

: "${TASK_NAME:?}" "${SEED:?}" "${OUTPUT_DIR:?}" "${QWEN_MEMORY_DIR:?}"
: "${GPT_MEMORY_FILE:?}" "${GPU:?}" "${ROBOTWIN_ASSETS_PATH:?}"
: "${INFINI_BASE_URL:?}" "${INFINI_API_KEY:?}" "${RPENT_BASE_URL:?}"
: "${RPENT_API_KEY:?}" "${RPENT_ESCALATION_PLAN:?}"
: "${RPENT_ESCALATION_CHECKIN:?}" "${RPENT_ESCALATION_ERRORS:?}"

v1() {
  local url="${1%/}"
  case "$url" in
    */v1) printf '%s' "$url" ;;
    *) printf '%s/v1' "$url" ;;
  esac
}

RPENT_BIN="${RPENT_BIN:-$(command -v rpent || true)}"
if [ -z "$RPENT_BIN" ] && [ -x "$ROOT/.runtime/venv/bin/rpent" ]; then
  RPENT_BIN="$ROOT/.runtime/venv/bin/rpent"
fi
: "${RPENT_BIN:?rpent entrypoint not found; set RPENT_BIN or activate an environment}"

unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 PYDANTIC_AI_NO_BANNER=1
export OPENAI_API_KEY="$INFINI_API_KEY"
export CELL_TIMEOUT_S="${SESSION_TIMEOUT_S:-1200}"
export RPENT_TOOL_TIMEOUT_S="${TOOL_TIMEOUT_S:-1200}"
export RPENT_COERCE_TOOL_ARGS=1
export RPENT_ESCALATION=1
export RPENT_ESCALATION_MODEL=openai:gpt-6-astra
export RPENT_ESCALATION_BASE_URL="$(v1 "$RPENT_BASE_URL")"
export RPENT_ESCALATION_KEY_ENV=RPENT_API_KEY
export RPENT_ESCALATION_REASONING_EFFORT=high
export RPENT_ESCALATION_MAX_SHARE="${RPENT_ESCALATION_MAX_SHARE:-0.2}"
export RPENT_CLOUD_CORRECTIONS="$GPT_MEMORY_FILE"

if [ -n "${VLA_ENDPOINT:-}" ]; then
  VLA_ARGS=(
    --vla-endpoint "$VLA_ENDPOINT"
    --env-cuda-device "$GPU"
  )
else
  : "${LINGBOT_MODEL_PATH:?}"
  VLA_ARGS=(
    --vla-model-path "$LINGBOT_MODEL_PATH"
    --cuda-device "$GPU"
  )
fi

mkdir -p "$OUTPUT_DIR"
exec "$RPENT_BIN" \
  --robot robotwin_compact \
  --task-name "$TASK_NAME" \
  --task-config demo_randomized \
  --seed "$SEED" \
  --planner api \
  --model openai-chat:qwen3.6-27b \
  --base-url "$(v1 "$INFINI_BASE_URL")" \
  --reasoning-effort none \
  --memory-profile local \
  --memory-dir "$QWEN_MEMORY_DIR" \
  --planner-timeout-s "${SESSION_TIMEOUT_S:-1200}" \
  --max-turns "${MAX_TURNS:-100}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-10000}" \
  --robotwin-assets-path "$ROBOTWIN_ASSETS_PATH" \
  "${VLA_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --verbose
