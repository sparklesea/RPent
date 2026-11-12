#!/usr/bin/env python3
"""Evaluate Qwen3.8-27B + GPT-6 RoboTwin collaboration with stage-2 rules."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_collab_eval as base  # noqa: E402


base.WORKER_MODEL = "openai-chat:qwen3.8-27b"
base.RUNNER = Path(__file__).with_name("run_collab_eval_qwen38_episode.sh")


if __name__ == "__main__":
    raise SystemExit(base.main())
