#!/usr/bin/env python3
"""RoboTwin event-triggered replanning baseline inspired by BRACE.

This is a deliberately small reproduction of BRACE's replanning controller.
Qwen executes most turns. GPT-6 is called for an initial plan and whenever a
dynamic trigger is admitted:

- consecutive tool errors;
- stalled RoboTwin end-effector motion;
- a periodic progress check-in.

The baseline does not implement E-RECAP, retrieval, dynamic token budgets,
latency SLOs, or explicit cooldown/commit windows. The existing escalation
wrapper still ignores tool results that the senior has already reviewed, which
prevents the same evidence from retriggering replanning every turn.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from pydantic_ai.messages import ModelMessage

from rpent.planner import escalation
from rpent.planner.escalation import EscalationModel, _results

DEFAULT_CHECKIN = 8
DEFAULT_ERRORS = 4
DEFAULT_STALL_WINDOW = 4
DEFAULT_STALL_SPAN_M = 0.01
DEFAULT_MAX_VETOES = 1


def _pose_points(result: dict) -> dict[str, list[float]]:
    """Return available RoboTwin end-effector positions from a tool result."""
    state = result.get("state")
    robot_state = state.get("robot_state") if isinstance(state, dict) else None
    if not isinstance(robot_state, dict):
        return {}

    points: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        value = robot_state.get(f"{arm}_eef_pose")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                match = re.search(r"\[\s*([-+0-9.eE\s]+)\]", value)
                value = match.group(1).split() if match else None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                points[arm] = [float(part) for part in value[:3]]
            except (TypeError, ValueError):
                continue
    return points


def _motion_span(frames: list[dict[str, list[float]]]) -> float | None:
    """Largest displacement across arms in a short pose window."""
    if not frames:
        return None
    common_arms = set(frames[0])
    for frame in frames[1:]:
        common_arms.intersection_update(frame)
    if not common_arms:
        return None
    return max(
        math.dist(frames[0][arm], frame[arm])
        for frame in frames[1:]
        for arm in common_arms
    )


class BRACEEventModel(EscalationModel):
    """Qwen worker with GPT replanning on dynamic execution events."""

    def __init__(
        self,
        worker,
        senior,
        *,
        checkin: int = DEFAULT_CHECKIN,
        errors: int = DEFAULT_ERRORS,
        stall_window: int = DEFAULT_STALL_WINDOW,
        stall_span_m: float = DEFAULT_STALL_SPAN_M,
        max_vetoes: int = DEFAULT_MAX_VETOES,
        **kwargs: Any,
    ) -> None:
        if stall_window < 2:
            raise ValueError("stall_window must be at least 2")
        if stall_span_m < 0:
            raise ValueError("stall_span_m must not be negative")
        super().__init__(
            worker,
            senior,
            checkin=checkin,
            errors=errors,
            max_share=1.0,
            max_vetoes=max_vetoes,
            plan=True,
            **kwargs,
        )
        self.stall_window = stall_window
        self.stall_span_m = stall_span_m

    def _trigger(self, messages: list[ModelMessage]) -> str | None:
        results = list(_results(messages))
        if not self.planned:
            names = {name for name, _ in results}
            if {
                "read_text_file",
                "view_env_state",
            } <= names or self.requests + 1 >= escalation.PLAN_FALLBACK_REQUEST:
                return "plan"
            return None
        if self.senior_requests >= self.max_share * (self.requests + 1):
            return None

        fresh = results[self.reviewed_results :]
        if len(fresh) >= self.errors and all(
            "error" in result for _, result in fresh[-self.errors :]
        ):
            return "tool_errors"

        poses = [
            points
            for _, result in fresh
            if (points := _pose_points(result))
        ]
        if len(poses) >= self.stall_window:
            span = _motion_span(poses[-self.stall_window :])
            if span is not None and span < self.stall_span_m:
                return "eef_stall"

        if self.since_senior >= self.checkin:
            return "periodic_checkin"
        return None

    def _senior_note(self, reason: str, proposal: dict | None = None) -> str:
        why = {
            "plan": "the task memory and initial scene are ready; set the plan",
            "tool_errors": "several consecutive tool calls returned errors",
            "eef_stall": "the robot end effectors barely moved over the recent steps",
            "periodic_checkin": "a periodic progress check-in is due",
        }.get(reason)
        if why is None:
            return super()._senior_note(reason, proposal)

        note = (
            "[harness] You are the senior planner in an event-triggered BRACE-style "
            f"baseline. A faster worker runs most turns; you take this turn because "
            f"{why}. Review the latest images and state, state a corrected plan in a "
            "few sentences, then make the next tool call yourself. Do not reply with "
            "text only."
        )
        if self.corrections:
            note += (
                "\nCloud correction memory for this task (your diagnoses and fixes "
                "from earlier episodes; apply an entry only when its situation "
                "matches):\n" + self.corrections
            )
        if proposal is not None:
            summary = str(proposal.get("summary", ""))[:1200]
            note += (
                f"\nThe worker proposed finish(status={proposal.get('status')!r}): "
                f"{summary}\nIf the task can still be completed, keep working; "
                "otherwise call finish yourself."
            )
        return note


_original_escalation_from_env = escalation.escalation_from_env


def brace_from_env(worker, *, output_dir, reasoning_effort):
    """Build the normal escalation model, then enable BRACE event triggers."""
    base_model = _original_escalation_from_env(
        worker, output_dir=output_dir, reasoning_effort=reasoning_effort
    )
    if os.environ.get("RPENT_BRACE") != "1":
        return base_model
    if not isinstance(base_model, EscalationModel):
        raise RuntimeError(
            "RPENT_BRACE=1 requires RPENT_ESCALATION=1 and an EscalationModel worker"
        )
    return BRACEEventModel(
        base_model.wrapped,
        base_model.senior,
        worker_settings=base_model.worker_settings,
        senior_settings=base_model.senior_settings,
        corrections=base_model.corrections,
        log_path=base_model.log_path,
        checkin=int(os.environ.get("RPENT_BRACE_CHECKIN", str(DEFAULT_CHECKIN))),
        errors=int(os.environ.get("RPENT_BRACE_ERRORS", str(DEFAULT_ERRORS))),
        stall_window=int(
            os.environ.get("RPENT_BRACE_STALL_WINDOW", str(DEFAULT_STALL_WINDOW))
        ),
        stall_span_m=float(
            os.environ.get("RPENT_BRACE_STALL_SPAN_M", str(DEFAULT_STALL_SPAN_M))
        ),
        max_vetoes=int(
            os.environ.get("RPENT_BRACE_MAX_VETOES", str(DEFAULT_MAX_VETOES))
        ),
    )


def main() -> int:
    escalation.escalation_from_env = brace_from_env
    from rpent.cli.main import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
