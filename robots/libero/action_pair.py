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

"""Generic two-primitive executor for open-loop and guarded pair experiments."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

# These are state-changing primitives that are safe to invoke through this
# experiment tool. Read-only perception tools are deliberately excluded.
PAIR_ACTIONS = (
    "move_to",
    "move_pose",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
)


def _validate_spec(spec: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"{label} must be an object")
    action = spec.get("action")
    if action not in PAIR_ACTIONS:
        raise ValueError(f"{label}.action must be one of {PAIR_ACTIONS}")
    args = spec.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"{label}.args must be an object")
    if set(spec) - {"action", "args"}:
        raise ValueError(f"{label} accepts only action and args")
    # Validate both primitives completely before either can change the scene.
    from robots.libero.tools import TOOLS_SPEC

    schema = next(s["input_schema"] for s in TOOLS_SPEC if s["name"] == action)
    error = next(
        Draft202012Validator({**schema, "additionalProperties": False}).iter_errors(
            args
        ),
        None,
    )
    if error is not None:
        raise ValueError(f"{label}.args: {error.message}")
    if "xyz" in args:
        xyz = np.asarray(args["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{label}.args.xyz must contain three finite numbers")
    if "gripper" in args:
        g = args["gripper"]
        if isinstance(g, bool) or float(g) not in (-1.0, 1.0):
            raise ValueError(f"{label}.args.gripper must be -1 or +1")
    return {"action": action, "args": dict(args)}


def _run(primitives: Any, spec: dict[str, Any]) -> dict[str, Any]:
    handler = getattr(primitives, spec["action"], None)
    if handler is None or spec["action"].startswith("_"):
        raise ValueError(f"primitive is unavailable: {spec['action']}")
    return handler(**spec["args"])


def _position_guard(result: dict[str, Any], max_error: float) -> bool:
    value = result.get("final_dist_m")
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) <= max_error
    )


def _guard_first(
    primitives: Any,
    first: dict[str, Any],
    result: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    active = not (primitives.env.terminated or primitives.env.truncated)
    action = first["action"]
    max_error = float(guard.get("max_position_error_m", 0.02))
    if not math.isfinite(max_error) or not 0.001 <= max_error <= 0.05:
        raise ValueError("max_position_error_m must be between 0.001 and 0.05")

    checks: dict[str, Any] = {"active": active}
    if action in {"move_to", "move_pose"}:
        checks["position_ok"] = _position_guard(result, max_error)
    elif action in {"rotate_wrist", "rotate_pitch"}:
        value = result.get("final_err")
        checks["orientation_ok"] = (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) <= float(guard.get("max_orientation_error_rad", 0.05))
        )
    elif action == "pi0_pick":
        # This is the pick heuristic, not proof of object identity or a secure hold.
        checks["action_success"] = result.get("success") is True
    elif action == "release":
        opening = result.get("final_gripper_opening")
        checks["gripper_open_ok"] = isinstance(opening, (int, float)) and float(
            opening
        ) >= float(guard.get("min_gripper_opening_m", 0.04))
    elif action == "set_gripper":
        opening = getattr(primitives, "_last_obs_gripper", None)
        target = float(first["args"].get("gripper", -1.0))
        checks["gripper_state_ok"] = isinstance(opening, (int, float)) and (
            (target < 0 and opening >= 0.04) or (target > 0 and opening < 0.06)
        )
    elif action == "pi0_doubled":
        # Contact skills can be intermediate. They must opt into open-loop
        # semantics unless they return an explicit success predicate.
        checks["action_success"] = result.get("success") is True
    checks["passed"] = bool(
        active and all(v is True for k, v in checks.items() if k != "active")
    )
    if not checks["passed"] and active and len(checks) == 1:
        checks["passed"] = bool(guard.get("allow_unverified_first", False))
    return checks


def execute_action_pair(
    primitives: Any,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    mode: str = "guarded",
    guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute two state-changing primitives in one tool call.

    open_loop skips all intermediate success checks and executes the second
    primitive whenever the episode remains active. guarded uses an
    action-aware local postcondition and skips the second on failure.
    """
    if mode not in {"open_loop", "guarded"}:
        raise ValueError("mode must be open_loop or guarded")
    first = _validate_spec(first, label="first")
    second = _validate_spec(second, label="second")
    guard = {} if guard is None else dict(guard)

    if primitives.env.terminated or primitives.env.truncated:
        return {
            "name": "action_pair",
            "mode": mode,
            "actions_executed": 0,
            "second_executed": False,
            "skip_reason": "episode_already_done",
            "terminated": primitives.env.terminated,
            "truncated": primitives.env.truncated,
        }

    started = time.perf_counter()
    first_result = _run(primitives, first)
    first_duration = time.perf_counter() - started
    active = not (primitives.env.terminated or primitives.env.truncated)
    if mode == "open_loop":
        checks = {"mode": "open_loop", "active_after_first": active, "passed": active}
    else:
        checks = _guard_first(primitives, first, first_result, guard)
    output = {
        "name": "action_pair",
        "mode": mode,
        "first": first_result,
        "first_duration_s": first_duration,
        "guard": checks,
        "actions_executed": 1,
        "second_executed": False,
    }
    if active and checks.get("passed") is True:
        primitives._check_cancelled()
        started = time.perf_counter()
        output["second"] = _run(primitives, second)
        output["second_duration_s"] = time.perf_counter() - started
        output["actions_executed"] = 2
        output["second_executed"] = True
    else:
        output["skip_reason"] = (
            "first_postcondition_failed_replan_from_latest_state"
            if mode == "guarded"
            else "episode_ended_after_first"
        )
    output.update(
        terminated=primitives.env.terminated, truncated=primitives.env.truncated
    )
    return output
