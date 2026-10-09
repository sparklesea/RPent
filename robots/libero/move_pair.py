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

"""Experimental two-translation executor with an intermediate state guard."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def execute_move_pair(
    primitives: Any,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    max_position_error_m: float = 0.02,
) -> dict[str, Any]:
    """Validate both commands before motion, then conditionally run the second."""
    allowed = {"xyz", "gripper", "step_clip", "max_steps", "tol"}
    position_limit = float(max_position_error_m)
    if not math.isfinite(position_limit) or not 0.001 <= position_limit <= 0.03:
        raise ValueError("position guard must be between 0.001 and 0.03 m")
    actions = []
    for supplied in (first, second):
        if not isinstance(supplied, dict) or set(supplied) - allowed:
            raise ValueError(
                "pair actions accept only xyz/gripper/step_clip/max_steps/tol"
            )
        xyz = np.asarray(supplied.get("xyz"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("each xyz must contain three finite numbers")
        gripper = supplied.get("gripper")
        if isinstance(gripper, bool) or gripper not in (-1, 1):
            raise ValueError("each gripper must explicitly be -1 or +1")
        action = dict(supplied, xyz=xyz.tolist())
        if "max_steps" in action:
            n = action["max_steps"]
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 150:
                raise ValueError("max_steps must be an integer from 1 to 150")
        for name, low, high in [("step_clip", 0.001, 0.025), ("tol", 0.001, 0.02)]:
            if name in action:
                value = float(action[name])
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{name} must be between {low} and {high}")
                action[name] = value
        actions.append(action)
    if actions[0]["gripper"] != actions[1]["gripper"]:
        raise ValueError("a pair must preserve the gripper command")
    if primitives.env.terminated or primitives.env.truncated:
        return {
            "name": "move_to_pair",
            "second_executed": False,
            "actions_executed": 0,
            "skip_reason": "episode_already_done",
            "terminated": primitives.env.terminated,
            "truncated": primitives.env.truncated,
        }
    # Preflight the first and nominal second traversal before changing the env.
    for source, target in [
        (primitives._last_obs_eef_pos, actions[0]["xyz"]),
        (actions[0]["xyz"], actions[1]["xyz"]),
    ]:
        if np.linalg.norm(np.asarray(target)[:2] - np.asarray(source)[:2]) > 0.30:
            raise ValueError("each pair xy traversal must be <=0.30 m")
    gap_before = float(primitives._last_obs_gripper)
    started = time.perf_counter()
    first_result = primitives.move_to(**actions[0])
    first_s = time.perf_counter() - started
    pos = np.asarray(primitives._last_obs_eef_pos, dtype=float)
    position_error = float(np.linalg.norm(pos - actions[0]["xyz"]))
    gap_after = float(primitives._last_obs_gripper)
    active = not (primitives.env.terminated or primitives.env.truncated)
    position_ok = math.isfinite(position_error) and position_error <= position_limit
    gap_ok = math.isfinite(gap_after) and abs(gap_after - gap_before) <= 0.01
    path_ok = float(np.linalg.norm(pos[:2] - np.asarray(actions[1]["xyz"])[:2])) <= 0.30
    guard = {
        "active": active,
        "position_ok": position_ok,
        "gripper_ok": gap_ok,
        "second_path_ok": path_ok,
        "observed_xyz": pos.tolist(),
        "position_error_m": position_error,
        "max_position_error_m": position_limit,
        "gripper_gap_before": gap_before,
        "gripper_gap_after": gap_after,
        "max_gripper_change_m": 0.01,
    }
    result = {
        "name": "move_to_pair",
        "first": first_result,
        "first_duration_s": first_s,
        "guard": guard,
        "actions_executed": 1,
        "second_executed": False,
    }
    if active and position_ok and gap_ok and path_ok:
        primitives._check_cancelled()
        started = time.perf_counter()
        result["second"] = primitives.move_to(**actions[1])
        result["second_duration_s"] = time.perf_counter() - started
        result.update(actions_executed=2, second_executed=True)
    else:
        result["skip_reason"] = "guard_failed_replan_from_latest_state"
    result.update(
        terminated=primitives.env.terminated, truncated=primitives.env.truncated
    )
    return result
