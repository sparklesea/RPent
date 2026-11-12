#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the compact-tool RoboTwin evaluation with the K8/K4 rule set."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

base = importlib.import_module("run_collab_eval")
_run_episode = base.run_episode


def _run_compact_episode(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Annotate each episode result with the compact-tool variant."""
    row = _run_episode(*args, **kwargs)
    row["tool_variant"] = "compact_memory_ref_chunks"
    run = args[0]
    run_dir = kwargs["run_dir"]
    output_dir = run_dir / (
        f"{run['task_name']}_entry{run['entry_index']}_s{run['seed']}"
    )
    base.write_json(output_dir / "result.json", row)
    return row


def main() -> int:
    """Run the existing collaborative evaluator with the compact episode runner."""
    base.RUNNER = HERE / "run_compact_eval_episode.sh"
    base.run_episode = _run_compact_episode
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
