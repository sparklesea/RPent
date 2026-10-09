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

"""LIBERO prompt bundle assembly."""

from __future__ import annotations

import os
from collections.abc import Mapping

from robots.libero.prompts import compact as compact_parts
from robots.libero.prompts import evaluate as evaluate_parts
from robots.libero.prompts import explore as explore_parts
from robots.libero.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    if (variables or {}).get("prompt_profile", "full") == "compact":
        prompt = compact_parts.SYSTEM
        pair_mode = os.getenv("RPENT_PAIR_MODE")
        if pair_mode in {"open_loop", "guarded"}:
            prompt = dict(prompt)
            prompt["DECISION LOOP"] = compact_parts.SYSTEM["DECISION LOOP"].replace(
                "Issue one motion-changing tool\n    at a time;",
                "Issue a robot action pair through action_pair; visual tools "
                "remain separate;",
            )
            prompt["ACTION PAIR"] = (
                f"STRICT ACTION PAIR MODE ({pair_mode}). For every planned "
                "robot-action sequence, use action_pair with exactly two "
                "state-changing primitives in first and second. Each primitive "
                "has an action name and args object. Allowed examples include "
                "move_to, pi0_pick, release, set_gripper, move_pose, "
                "rotate_wrist, rotate_pitch, and pi0_doubled. You may pair "
                "move_to with pi0_pick or release when the second action is "
                "the intended next operation; do not put segment, "
                "back_project, view_env_state, or other read-only tools inside "
                "the pair. In open_loop mode the executor runs second whenever "
                "the episode is still active. In guarded mode it checks the "
                "first primitive's local result and robot state, and skips "
                "second if the postcondition fails. The check is local and "
                "does not consume an LLM request. Do not call direct move_to "
                "or move_pose in this strict mode; submit them through "
                "action_pair. A failed guarded pair must be replanned from "
                "the returned latest state."
            )
            return prompt
        if os.getenv("RPENT_ENABLE_MOVE_PAIR") == "1":
            prompt = dict(prompt)
            prompt["DECISION LOOP"] = compact_parts.SYSTEM["DECISION LOOP"].replace(
                "Issue one motion-changing tool\n    at a time;",
                "Issue one motion-changing tool at a time (move_to_pair counts "
                "as one guarded tool);",
            )
            prompt["BATCHED MOTION"] = (
                "When two successive translations have already known targets, "
                "use move_to_pair instead of asking the LLM again between them. "
                "Examples: carry midpoint then above destination; raise the "
                "held object vertically then traverse to an above-destination "
                "waypoint. Supply first and second as {xyz:[x,y,z],gripper:1} "
                "(or -1 for both), optionally step_clip, tol, max_steps. "
                "The executor uses actual EEF position and actual gripper gap "
                "after the first; it skips the second on termination/truncation, "
                "position error >0.02 m or gap change >0.01 m. It returns final "
                "images for the next LLM decision. Prefer a pair whenever both "
                "translations are known; never invent a redundant waypoint "
                "just to make a pair. Keep each xy traversal <=0.30 m. Never "
                "batch across picking/release/contact/reorientation or when "
                "the next move requires visual grasp or object verification. "
                "These checks do not certify a held object's identity or detect "
                "slip reliably, so only batch previously confirmed free-space "
                "carrying or open-gripper preposition moves."
            )
        return prompt
    return evaluate_parts.system_prompt(variables)


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": (
            compact_parts.BEGIN
            if (variables or {}).get("mode", "eval") == "eval"
            and (variables or {}).get("prompt_profile", "full") == "compact"
            else user_parts.BEGIN
        ),
    }


__all__ = ["system_prompt", "user_prompt"]
