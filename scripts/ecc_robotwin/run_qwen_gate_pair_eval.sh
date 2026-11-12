#!/usr/bin/env bash
# RoboTwin Qwen gate with GPT fallback for isolated tool-sequence choices.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export RPENT_CANDIDATE_GATE_MIN_TOOL_AGREEMENT="${RPENT_CANDIDATE_GATE_MIN_TOOL_AGREEMENT:-2}"
exec "$HERE/run_qwen_gate_eval.sh" \
  --gate-min-tool-agreement "$RPENT_CANDIDATE_GATE_MIN_TOOL_AGREEMENT" "$@"
