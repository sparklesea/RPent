#!/usr/bin/env python3
"""RoboTwin GPT-plan baseline with a timed midpoint replan.

This entrypoint keeps the existing RPent CLI and planner stack unchanged. It
patches only the opt-in escalation factory in this process, then delegates to
``rpent.cli.main`` with the normal arguments.

The worker is Qwen. GPT-6 takes the first planning turn. If the episode is still
running after ``replan_interval_s`` seconds (600 s by default), GPT-6 takes one
more planning turn; Qwen handles every other turn. The worker and senior use
their existing separate memory channels:

- Qwen memory comes from ``--memory-dir`` (``QWEN_MEMORY_DIR`` in the episode
  runner).
- GPT memory comes from ``RPENT_CLOUD_CORRECTIONS`` (``GPT_MEMORY_FILE``).
"""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic_ai.messages import ModelMessage

from rpent.planner import escalation
from rpent.planner.escalation import EscalationModel

TIME_REPLAN_REASON = "time_replan"
DEFAULT_REPLAN_INTERVAL_S = 600.0


class PlanTimedModel(EscalationModel):
    """Qwen worker with GPT planning and a timed midpoint replan."""

    def __init__(
        self,
        worker,
        senior,
        *,
        replan_interval_s: float = DEFAULT_REPLAN_INTERVAL_S,
        **kwargs: Any,
    ) -> None:
        if replan_interval_s < 0:
            raise ValueError("replan_interval_s must not be negative")
        # Disable the generic check-in, error, and give-up escalation paths.
        super().__init__(
            worker,
            senior,
            checkin=10**9,
            errors=10**9,
            max_share=1.0,
            max_vetoes=0,
            plan=True,
            **kwargs,
        )
        self.replan_interval_s = replan_interval_s
        self._started_at = time.monotonic()
        self._next_time_replan = (
            self._started_at + replan_interval_s if replan_interval_s else None
        )

    def _trigger(self, messages: list[ModelMessage]) -> str | None:
        if not self.planned:
            return super()._trigger(messages)
        if (
            self._next_time_replan is not None
            and time.monotonic() >= self._next_time_replan
        ):
            self._next_time_replan = time.monotonic() + self.replan_interval_s
            return TIME_REPLAN_REASON
        return None

    def _senior_note(self, reason: str, proposal: dict | None = None) -> str:
        if reason != TIME_REPLAN_REASON:
            return super()._senior_note(reason, proposal)
        note = (
            "[harness] You are the senior planner. A faster worker model runs "
            "most turns of this shared episode; you take this turn because "
            f"{self.replan_interval_s:.0f} seconds of the episode have elapsed "
            "without completion. Review the latest images and state, state a "
            "corrected plan in a few sentences for the worker, then make the "
            "next tool call yourself. Do not reply with text only."
        )
        if self.corrections:
            note += (
                "\nCloud correction memory for this task (your diagnoses and "
                "fixes from earlier episodes; apply an entry only when its "
                "situation matches):\n" + self.corrections
            )
        return note


_original_escalation_from_env = escalation.escalation_from_env


def plan_timed_from_env(worker, *, output_dir, reasoning_effort):
    """Build the normal escalation model, then switch it to timed-only mode."""
    base_model = _original_escalation_from_env(
        worker, output_dir=output_dir, reasoning_effort=reasoning_effort
    )
    if os.environ.get("RPENT_PLAN_TIMED") != "1":
        return base_model
    if not isinstance(base_model, EscalationModel):
        raise RuntimeError(
            "RPENT_PLAN_TIMED=1 requires RPENT_ESCALATION=1 and an "
            "EscalationModel worker"
        )
    replan_interval_s = float(
        os.environ.get("RPENT_PLAN_TIME_REPLAN_S", str(DEFAULT_REPLAN_INTERVAL_S))
    )
    return PlanTimedModel(
        base_model.wrapped,
        base_model.senior,
        worker_settings=base_model.worker_settings,
        senior_settings=base_model.senior_settings,
        corrections=base_model.corrections,
        log_path=base_model.log_path,
        replan_interval_s=replan_interval_s,
    )


def main() -> int:
    escalation.escalation_from_env = plan_timed_from_env
    from rpent.cli.main import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
