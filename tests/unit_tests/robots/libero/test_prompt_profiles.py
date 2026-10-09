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

from robots.libero.prompt_bundle import system_prompt
from rpent.prompt.utils import format_prompt

_VARIABLES = {
    "suite": "libero_object_task",
    "task": 1,
    "seed": 0,
    "recipe_tag": "object_task_t1_s0",
    "mode": "eval",
    "memory_profile": "hf",
    "memory_dir": "/tmp/memory/libero",
    "reference_tag": "object_task_t1_s0",
    "memory_inbox": "/tmp/memory/inbox",
    "session_number": 1,
    "session_max": 1,
    "output_dir": "/tmp/output",
}


def _render(prompt_profile: str | None) -> str:
    variables = dict(_VARIABLES)
    if prompt_profile is not None:
        variables["prompt_profile"] = prompt_profile
    return format_prompt(system_prompt(variables), variables=variables)


def test_compact_profile_keeps_safety_critical_protocol_and_is_smaller():
    full = _render(None)
    compact = _render("compact")

    assert len(compact) < len(full) * 0.25
    assert "/task-specific/" in compact
    assert "/task_only/" not in compact
    for required in (
        "task_language",
        "Do not call reset",
        "back_project",
        "pi0_pick",
        "pi0_doubled",
        "gripper:1",
        "terminated",
        "finish",
    ):
        assert required in compact


def test_prompt_profile_defaults_to_full():
    assert _render(None) == _render("full")
