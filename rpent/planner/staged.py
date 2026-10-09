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

"""Phase planning, local execution/verification, and bounded repair for LIBERO."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelRetry,
    Tool,
    ToolReturn,
    capture_run_messages,
)
from pydantic_ai.capabilities import Thinking
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import RetryPromptPart, ToolCallPart
from pydantic_ai.models import Model

from rpent.dashboard.events import DashboardEventSink, TranscriptEvent
from rpent.planner.accounting import RequestAccounting, utc_now
from rpent.planner.api_loop import _build_model_settings
from rpent.planner.base import REASONING_EFFORTS, Planner, PlannerResult
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.logging import get_logger

CONTEXT_RULES = """phase_step is a workflow index, not an environment observation
index. For current perception use environment_step or step=-1. Reuse the
supplied memory_reads instead of rereading the same files. perception_evidence
contains prior observations with their environment step: reuse stationary
geometry, but re-ground moved objects. Check supervisor identity and measured
geometry against low-confidence segment results; do not silently replace them.
"""

DecisionT = TypeVar("DecisionT", bound=BaseModel)

logger = get_logger("staged")

# File writes, finish, reset and compound actions cannot bypass step verification.
READ_TOOLS = {
    "view_env_state",
    "view_camera_meta",
    "segment",
    "back_project",
    "read_text_file",
    "list_dir",
}
ACTION_TOOLS = {
    "move_to",
    "move_pose",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
}

SUPERVISOR = """You are the phase supervisor. Produce ONE useful phase with 1-5
concrete small steps, each with observable acceptance criteria. Name the actual
object/destination, grasp or contact instruction, movement intent, and necessary
geometry. Use images and perception tools; no GT object poses. The executor
will ground exact tool arguments again against the current scene. Do not plan
an entire long trajectory using stale coordinates. Do not include reset,
finish, audit writes, or compound action pairs. Each small step uses ONE robot
primitive followed by executor verification. Perception tools remain separate.
On a handoff, preserve confirmed completed work and use the failed checks and
repair history to change the remaining approach. Keep reasoning concise.
The runtime owns audit and finish; native terminated is the only task success.
"""
EXECUTOR = """You execute one small step of the supervisor's phase. Use current
images/state, relevant memory and perception tools to ground ONE primitive.
Return tool and arguments_json (a JSON object encoded as a string). Only choose
an allowed action from the supplied schemas. Do not execute any motion through
perception tools. For pi0_pick use a short grounded grasp instruction, not the
full task. Re-ground moved objects; retain correct object identity. A repair
must address the reported failure with a meaningful change, not repeat an
identical unsuccessful command. Keep the reason to one short sentence.
The runtime owns audit and finish; ignore conflicting instructions about them.
"""
VERIFIER = """Check the acceptance criteria for ONE small step from the newest
images, measured robot state, and tool results. Return PASS, FAIL or UNKNOWN
with concrete evidence. Use perception tools if needed. PASS requires evidence
for EVERY criterion. If occluded or ambiguous return UNKNOWN, not a guess.
EEF position/gripper gap alone cannot prove the right object is held or placed.
pi0_pick.success is a heuristic; pi0_doubled.success concerns the WHOLE task,
not intermediate contact success. An open gripper does not prove placement.
Never issue robot actions. Keep the evidence concise. Native terminated is
supplied separately by the runtime and alone determines whole-task success.
"""


class PhaseStep(BaseModel):
    """A concrete primitive-sized instruction and its observable checks."""

    instruction: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=5)


class PhasePlan(BaseModel):
    """One bounded subgoal, replanned from the current observation."""

    goal: str = Field(min_length=1)
    steps: list[PhaseStep] = Field(min_length=1, max_length=5)


class ActionRequest(BaseModel):
    """Runtime-validated choice of one registered robot primitive."""

    tool: str
    arguments_json: str
    reason: str

    @model_validator(mode="after")
    def validate_action(self):
        if self.tool not in ACTION_TOOLS:
            raise ValueError(f"not an individual robot primitive: {self.tool}")
        if not isinstance(json.loads(self.arguments_json), dict):
            raise ValueError("arguments_json must encode an object")
        return self


class StepAssessment(BaseModel):
    """Executor verification; UNKNOWN consumes the same repair budget as FAIL."""

    verdict: Literal["PASS", "FAIL", "UNKNOWN"]
    evidence: str = Field(min_length=1)


class StagedPlanner(Planner):
    """Give phase planning to one model and execution/checks to another.

    All requests and tools share one wall-clock deadline and request budget.
    Each step gets one initial action, then one local repair round with at most
    ``repair_actions`` actions. Every action is followed by executor verification.
    """

    def __init__(
        self,
        *,
        supervisor_model: Model,
        executor_model: Model,
        dashboard_events: DashboardEventSink,
        output_dir: str | Path,
        recipe_tag: str,
        max_tokens: int = 8192,
        timeout_s: float = 1200,
        reasoning_effort: str = "none",
        repair_actions: int = 3,
        perception_calls: int = 12,
    ):
        if (
            max_tokens < 1
            or timeout_s <= 0
            or repair_actions < 0
            or perception_calls < 1
        ):
            raise ValueError("invalid staged planner budget")
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        self.supervisor_model = supervisor_model
        self.executor_model = executor_model
        self.events = dashboard_events
        self.output_dir = Path(output_dir)
        self.recipe_tag = recipe_tag
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.repair_actions = repair_actions
        self.perception_calls = perception_calls

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction=None,
    ) -> PlannerResult:
        """Run a non-interactive LIBERO evaluation and retain partial accounting."""
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError(
                "staged currently supports non-interactive evaluation only"
            )
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        session = _StagedSession(self, toolkit, system_prompt, user_message, max_turns)
        return asyncio.run(session.run())


class _StagedSession:
    """Own one task's deadline, evidence, tools, and transcripts."""

    def __init__(
        self,
        planner: StagedPlanner,
        toolkit: Toolkit,
        system_prompt: str,
        task: str,
        max_turns: int,
    ) -> None:
        self.planner = planner
        self.toolkit = toolkit
        self.system_prompt = system_prompt
        self.task = task
        self.max_turns = max_turns
        self.specs = {spec["name"]: spec for spec in toolkit.get_tools_spec()}
        if "view_env_state" not in self.specs:
            raise ValueError("staged requires LIBERO's view_env_state tool")
        self.validators = {
            name: Draft202012Validator(spec["input_schema"])
            for name, spec in self.specs.items()
        }
        self.requests: list[dict] = []
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        self.last_observation: ToolResult | None = None
        self.last_action: dict | None = None
        self.memory_reads: dict[str, dict] = {}
        self.perception_evidence: list[dict] = []
        self.completed: list[dict] = []
        self.handoff: dict | None = None
        self.phase = 0
        self.step = 0
        self.perception_used = 0

    def emit(self, kind: str, **values: Any) -> None:
        record = {"type": kind, "phase": self.phase, "step": self.step, **values}
        self.messages.append(record)
        self.planner.events.emit(TranscriptEvent(record))
        logger.info("[%s] %s", kind, json.dumps(values, ensure_ascii=False))

    async def execute(self, name: str, arguments: dict, *, role: str) -> ToolResult:
        error = next(self.validators[name].iter_errors(arguments), None)
        if error is not None:
            raise ModelRetry(f"Invalid arguments for {name}: {error.message}")
        record = {
            "tool": name,
            "args": arguments,
            "role": role,
            "phase": self.phase,
            "step": self.step,
            "started_at": utc_now(),
        }
        self.tools.append(record)
        self.emit("tool_call", role=role, tool=name, args=arguments)
        started = time.perf_counter()
        operation = asyncio.create_task(
            asyncio.to_thread(self.toolkit.execute_tool, name, arguments)
        )
        try:
            result = await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Cancellation of the asyncio waiter does not stop physical work.
            await asyncio.to_thread(self.toolkit.cancel_active_and_wait)
            await operation
            record["cancelled"] = True
            raise
        finally:
            record.update(
                finished_at=utc_now(), duration_s=time.perf_counter() - started
            )
        self.emit("tool_result", role=role, tool=name, content=self.text(result))
        if name == "read_text_file" and not result.result.get("error"):
            self.memory_reads[arguments["path"]] = {
                "path": arguments["path"],
                "text": self.text(result),
            }
        elif name in {"segment", "back_project", "view_camera_meta"}:
            self.perception_evidence.append(
                {"tool": name, "arguments": arguments, "result": self.text(result)}
            )
        if (name == "view_env_state" and role == "runtime") or name in ACTION_TOOLS:
            self.last_observation = result
        if name in ACTION_TOOLS:
            self.last_action = {
                "tool": name,
                "arguments": arguments,
                "result": self.text(result),
            }
        return result

    @staticmethod
    def text(result: ToolResult) -> str:
        return "\n".join(
            b["text"] for b in result.content_blocks if b["type"] == "text"
        )

    @staticmethod
    def content(result: ToolResult) -> list[str | BinaryContent]:
        out = []
        for block in result.content_blocks:
            if block["type"] == "text":
                out.append(block["text"])
            elif block["type"] == "image":
                source = block["source"]
                out.append(
                    BinaryContent(
                        data=base64.b64decode(source["data"]),
                        media_type=source["media_type"],
                    )
                )
        return out

    async def perceive(self, name: str, **arguments: Any) -> ToolReturn:
        if self.perception_used >= self.planner.perception_calls:
            raise ModelRetry(
                "Perception budget reached; return your structured decision"
            )
        self.perception_used += 1
        return ToolReturn(
            return_value=self.content(
                await self.execute(name, arguments, role=self.perception_role)
            )
        )

    async def ask(
        self, role: str, output_type: type[DecisionT], instructions: str, context: dict
    ) -> DecisionT:
        model = (
            self.planner.supervisor_model
            if role == "supervisor"
            else self.planner.executor_model
        )
        self.perception_used = 0
        self.perception_role = role
        tools = [
            Tool.from_schema(
                partial(self.perceive, name),
                name=name,
                description=spec.get("description"),
                json_schema=spec["input_schema"],
                takes_ctx=False,
                sequential=True,
            )
            for name, spec in self.specs.items()
            if name in READ_TOOLS
        ]
        agent = Agent(
            model,
            output_type=output_type,
            tools=tools,
            system_prompt=(
                self.system_prompt
                + "\n\nROLE OVERRIDE:\n"
                + instructions
                + CONTEXT_RULES
            ),
            model_settings=_build_model_settings(model, self.planner.max_tokens),
            retries=2,
            capabilities=[
                Thinking(
                    False
                    if self.planner.reasoning_effort == "none"
                    else self.planner.reasoning_effort
                ),
                RequestAccounting(self.requests, role=role, limit=self.max_turns),
            ],
        )
        prompt = [
            json.dumps(
                {
                    "task": self.task,
                    "phase": self.phase,
                    "phase_step": self.step,
                    "environment_step": (
                        self.last_observation.result.get("step", -1)
                        if self.last_observation
                        else -1
                    ),
                    "last_action": self.last_action,
                    "memory_reads": list(self.memory_reads.values()),
                    "perception_evidence": self.perception_evidence[-20:],
                    **context,
                },
                ensure_ascii=False,
            )
        ]
        if self.last_observation is not None:
            prompt.extend(self.content(self.last_observation))
        if output_type is ActionRequest:

            @agent.output_validator
            def validate_action(output: ActionRequest) -> ActionRequest:
                if output.tool not in self.validators:
                    raise ModelRetry(f"primitive unavailable: {output.tool}")
                error = next(
                    self.validators[output.tool].iter_errors(
                        json.loads(output.arguments_json)
                    ),
                    None,
                )
                if error is not None:
                    raise ModelRetry(
                        f"Invalid arguments for {output.tool}: {error.message}"
                    )
                return output

        with capture_run_messages() as sdk_messages:
            try:
                result = await agent.run(prompt)
            finally:
                for message in sdk_messages:
                    for part in message.parts:
                        if isinstance(part, RetryPromptPart):
                            self.emit(
                                "validation_retry",
                                role=role,
                                tool=part.tool_name,
                                feedback=part.model_response(),
                            )
                        elif isinstance(part, ToolCallPart):
                            self.emit(
                                "model_tool_call",
                                role=role,
                                tool=part.tool_name,
                                arguments=part.args,
                            )
        output = result.output
        self.emit("decision", role=role, output=output.model_dump())
        return output

    def ended(self) -> bool:
        return self.toolkit.solved() or bool(
            self.last_observation and self.last_observation.result.get("truncated")
        )

    async def work(self) -> None:
        await self.execute("view_env_state", {"step": -1}, role="runtime")
        if self.last_observation.result.get("error"):
            raise RuntimeError(
                "initial observation unavailable: " + self.text(self.last_observation)
            )
        self.task = self.last_observation.result.get("task_language") or self.task
        while not self.ended():
            self.phase += 1
            self.step = 0
            plan = await self.ask(
                "supervisor",
                PhasePlan,
                SUPERVISOR,
                {
                    "completed": self.completed[-20:],
                    "handoff": self.handoff,
                    "action_schemas": [
                        v for k, v in self.specs.items() if k in ACTION_TOOLS
                    ],
                },
            )
            self.handoff = None
            for index, step in enumerate(plan.steps, 1):
                self.step = index
                attempts = []
                passed = False
                for attempt in range(1 + self.planner.repair_actions):
                    attempted_action = None
                    try:
                        action = await self.ask(
                            "executor",
                            ActionRequest,
                            EXECUTOR,
                            {
                                "goal": plan.goal,
                                "phase_plan": plan.model_dump(),
                                "confirmed_completed": self.completed[-10:],
                                "instruction": step.model_dump(),
                                "repair": attempt > 0,
                                "prior_attempts": attempts,
                                "action_schemas": [
                                    v
                                    for k, v in self.specs.items()
                                    if k in ACTION_TOOLS
                                ],
                            },
                        )
                        arguments = json.loads(action.arguments_json)
                        if action.tool not in self.specs:
                            raise ValueError(f"primitive unavailable: {action.tool}")
                        result = await self.execute(
                            action.tool, arguments, role="executor"
                        )
                        attempted_action = self.last_action
                        if self.ended():
                            return
                        # Stateful LIBERO tools already return the captured observation.
                        assessment = await self.ask(
                            "verifier",
                            StepAssessment,
                            VERIFIER,
                            {
                                "goal": plan.goal,
                                "instruction": step.model_dump(),
                                "action": self.last_action,
                                "repair": attempt > 0,
                            },
                        )
                        raw = result.result
                        primitive_result = raw.get("log", {}).get("result", raw)
                        errored = raw.get("error") or primitive_result.get("error")
                    except UnexpectedModelBehavior as exc:
                        errored = str(exc)
                        assessment = StepAssessment(
                            verdict="UNKNOWN",
                            evidence="Decision validation failed: " + errored,
                        )
                        self.emit(
                            "decision_error", role="executor_or_verifier", error=errored
                        )
                    passed = assessment.verdict == "PASS" and not errored
                    attempts.append(
                        {
                            "action": attempted_action,
                            "assessment": assessment.model_dump(),
                            "tool_error": errored,
                        }
                    )
                    self.emit(
                        "step_check",
                        passed=passed,
                        repair=attempt > 0,
                        assessment=assessment.model_dump(),
                        tool_error=errored,
                    )
                    if passed:
                        self.completed.append(
                            {
                                "goal": plan.goal,
                                "step": step.instruction,
                                "evidence": assessment.evidence,
                            }
                        )
                        break
                    if attempt == 0 and self.planner.repair_actions:
                        self.emit("repair_start", budget=self.planner.repair_actions)
                if not passed:
                    self.handoff = {
                        "goal": plan.goal,
                        "failed_step": step.model_dump(),
                        "attempts": attempts,
                        "remaining_steps": [s.model_dump() for s in plan.steps[index:]],
                    }
                    self.emit("handoff", evidence=self.handoff)
                    break

    async def run(self) -> PlannerResult:
        started = time.perf_counter()
        error = None
        try:
            await asyncio.wait_for(self.work(), timeout=self.planner.timeout_s)
        except asyncio.TimeoutError:
            error = f"planner timed out after {self.planner.timeout_s:g}s"
        except UsageLimitExceeded as exc:
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Staged planner failed")
        finally:
            await asyncio.to_thread(self.toolkit.cancel_active_and_wait)
        solved = self.toolkit.solved()
        summary = "Native task success" if solved else (error or "Episode truncated")
        finish = {"status": "success" if solved else "failure", "summary": summary}
        stats = {
            "turns_used": len(self.requests),
            "tool_calls": len(self.tools),
            "total_input_tokens": sum(r.get("input_tokens", 0) for r in self.requests),
            "total_output_tokens": sum(
                r.get("output_tokens", 0) for r in self.requests
            ),
            "total_cached_input_tokens": sum(
                r.get("cache_read_tokens", 0) for r in self.requests
            ),
            "timing": {"model_requests": self.requests, "tools": self.tools},
            "agent_elapsed_s": time.perf_counter() - started,
            "by_role": {
                role: {
                    "requests": sum(r["role"] == role for r in self.requests),
                    "input_tokens": sum(
                        r.get("input_tokens", 0)
                        for r in self.requests
                        if r["role"] == role
                    ),
                    "output_tokens": sum(
                        r.get("output_tokens", 0)
                        for r in self.requests
                        if r["role"] == role
                    ),
                    "llm_api_s": sum(
                        r["duration_s"] for r in self.requests if r["role"] == role
                    ),
                }
                for role in ("supervisor", "executor", "verifier")
            },
            "staged": {
                "phases": self.phase,
                "repair_actions": self.planner.repair_actions,
                "timeout_s": self.planner.timeout_s,
                "max_tokens": self.planner.max_tokens,
                "supervisor_model": self.planner.supervisor_model.model_name,
                "executor_model": self.planner.executor_model.model_name,
            },
        }
        # Persist partial evidence for timeouts/provider errors as well as success.
        self.planner.output_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "regime": "strict_perception",
            "planner": "staged",
            "terminated": solved,
            "finish": finish,
            "error": error,
            "stats": stats,
            "events": self.messages,
        }
        (self.planner.output_dir / f"{self.planner.recipe_tag}_staged.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # No model can report whole-task success without the native predicate.
        if not error and "finish" in self.specs:
            await self.execute("finish", finish, role="runtime")
        stats["tool_calls"] = len(self.tools)
        stats["agent_elapsed_s"] = time.perf_counter() - started
        (self.planner.output_dir / f"{self.planner.recipe_tag}_staged.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return PlannerResult(
            finish_result=finish, messages=self.messages, stats=stats, error=error
        )
