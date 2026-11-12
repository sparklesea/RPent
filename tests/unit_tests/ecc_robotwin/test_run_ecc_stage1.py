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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "ecc_robotwin"
    / "run_ecc_stage1.py"
)


def _load_stage1():
    spec = importlib.util.spec_from_file_location("run_ecc_stage1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_status_accepts_a_runner_error_as_the_first_row(tmp_path) -> None:
    stage1 = _load_stage1()
    settings = {"planner_timeout_s": 1200, "tool_timeout_s": 900}

    stage1.update_status(
        tmp_path,
        [{"task": "click_bell", "solved": False, "runner_error": "boom"}],
        1,
        settings,
    )

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["planner_timeout_s"] == 1200
    assert status["tool_timeout_s"] == 900
    assert status["finished_count"] == 1
