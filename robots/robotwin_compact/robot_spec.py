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

"""RobotSpec and toolkit factory for compact RoboTwin evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from robots.robotwin import robot_spec as base
from robots.robotwin_compact import prompts
from robots.robotwin_compact.toolkit import CompactRoboTwinToolkit
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir


def get_robot_spec() -> RobotSpec:
    """Return the RoboTwin eval spec with compact prompts and tools."""
    return replace(
        base.get_robot_spec(),
        name="robotwin_compact",
        prompts=PromptBundle(system=prompts.system_prompt, user=prompts.user_prompt),
        supports_exploration=False,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
) -> CompactRoboTwinToolkit:
    """Build the compact RoboTwin eval toolkit."""
    if mode != "evaluation":
        raise ValueError("robotwin_compact supports evaluation mode only")
    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robotwin"),
        memory_access="read_only",
    )
    return CompactRoboTwinToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        reference_tag=str(config.prompt_vars["reference_tag"]),
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=state_output_dir,
    )
