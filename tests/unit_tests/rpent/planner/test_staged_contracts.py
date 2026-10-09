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

"""Exercise phase/repair/verification behavior through the installed Agent SDK."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import build_api_model, build_planner
from rpent.planner.staged import StagedPlanner
from rpent.tools.toolkit import Toolkit, readonly


class Scene(Toolkit):
    def __init__(self, *, solve_after=2, error=False, blocking=False):
        self.actions = []
        self.solve_after = solve_after
        self.error = error
        self.blocking = blocking
        self.started = threading.Event()
        self.stopped = threading.Event()
        super().__init__(
            dashboard_events=NullDashboardEventSink(), memory=SimpleNamespace()
        )

    def _register_common_tools(self):
        self.add_tool(
            "view_env_state",
            {
                "name": "view_env_state",
                "input_schema": {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                },
            },
            self.observe,
        )
        self.add_tool(
            "move_to",
            {
                "name": "move_to",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            },
            self.move,
        )

    @readonly
    def observe(self, step=-1):
        return {
            "task_language": "Put the mug in the basket",
            "step": len(self.actions),
            "state": {"x": len(self.actions)},
            "terminated": self.solved(),
            "truncated": False,
            "_image_bytes": b"fresh image",
        }

    @readonly
    def move(self, x):
        self.actions.append(x)
        if self.blocking:
            self.started.set()
            try:
                while True:
                    self.raise_if_cancelled()
                    time.sleep(0.005)
            finally:
                self.stopped.set()
        return {
            **self.observe(),
            "log": {"result": {"error": "stalled"} if self.error else {}},
        }

    def solved(self):
        return len(self.actions) >= self.solve_after and not self.error


@pytest.fixture(autouse=True)
def no_templates(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)


def phase(*instructions):
    return {
        "goal": "Place mug",
        "steps": [
            {
                "instruction": instruction,
                "acceptance_criteria": ["Mug visibly at target"],
            }
            for instruction in instructions
        ],
    }


def action(x):
    return {
        "tool": "move_to",
        "arguments_json": json.dumps({"x": x}),
        "reason": "Move mug",
    }


def check(verdict):
    return {"verdict": verdict, "evidence": "Observed target state"}


def scripted(outputs, calls):
    async def respond(messages, info):
        calls.append(messages)
        output = outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, output)],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    return FunctionModel(respond)


def solve(tmp_path, supervisor, executor, *, scene=None, max_turns=30, **kwargs):
    supervisor_calls, executor_calls = [], []
    scene = scene or Scene()
    planner = StagedPlanner(
        supervisor_model=scripted(supervisor, supervisor_calls),
        executor_model=scripted(executor, executor_calls),
        output_dir=tmp_path,
        recipe_tag="task",
        dashboard_events=NullDashboardEventSink(),
        **kwargs,
    )
    result = planner.solve(
        system_prompt="Use observations",
        user_message="task",
        toolkit=scene,
        max_turns=max_turns,
    )
    return result, scene, supervisor_calls, executor_calls


def test_phase_executes_in_order_and_only_native_success_counts(tmp_path):
    result, scene, supervisor, executor = solve(
        tmp_path, [phase("Carry", "Lower")], [action(1), check("PASS"), action(2)]
    )
    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert scene.actions == [1, 2]
    assert len(supervisor) == 1 and len(executor) == 3
    assert result.stats["turns_used"] == 4
    assert result.stats["total_input_tokens"] == 400
    assert [r["role"] for r in result.stats["timing"]["model_requests"]] == [
        "supervisor",
        "executor",
        "verifier",
        "executor",
    ]
    audit = json.loads((tmp_path / "task_staged.json").read_text())
    assert audit["terminated"] is True
    assert "fresh image" not in (tmp_path / "task_staged.json").read_text()


def test_unknown_repairs_locally_then_escalates_with_failed_checks(tmp_path):
    result, scene, supervisor, _ = solve(
        tmp_path,
        [phase("Carry"), phase("Change approach")],
        [action(1), check("UNKNOWN"), action(2), check("FAIL"), action(3)],
        scene=Scene(solve_after=3),
        repair_actions=1,
    )
    assert result.finish_result["status"] == "success"
    assert scene.actions == [1, 2, 3]
    assert len(supervisor) == 2
    handoff = next(m for m in result.messages if m["type"] == "handoff")
    assert len(handoff["evidence"]["attempts"]) == 2
    # The next supervisor receives repair evidence as well as the new image/state.
    assert "UNKNOWN" in str(supervisor[1]) and "FAIL" in str(supervisor[1])


def test_phase_handoff_preserves_read_evidence_and_environment_index(tmp_path):
    scene = Scene(solve_after=2)

    @readonly
    def read_memory(path):
        return {"path": path, "content": "Confirm blue mug identity"}

    @readonly
    def back_project(step):
        return {"step": step, "world_xyz": [0.1, 0.2, 0.3]}

    for name, handler, properties in [
        ("read_text_file", read_memory, {"path": {"type": "string"}}),
        ("back_project", back_project, {"step": {"type": "integer"}}),
    ]:
        scene.add_tool(
            name,
            {
                "name": name,
                "input_schema": {"type": "object", "properties": properties},
            },
            handler,
        )
    calls = []

    async def supervisor(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("read_text_file", {"path": "memory.md"}),
                    ToolCallPart("back_project", {"step": 0}),
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, phase("Carry"))]
        )

    executor_calls = []
    planner = StagedPlanner(
        supervisor_model=FunctionModel(supervisor),
        executor_model=scripted([action(1), check("PASS"), action(2)], executor_calls),
        output_dir=tmp_path,
        recipe_tag="task",
        dashboard_events=NullDashboardEventSink(),
    )
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=scene, max_turns=20
    )
    assert result.finish_result["status"] == "success"
    received = str(calls[2])
    assert "Confirm blue mug identity" in received and "world_xyz" in received
    assert '"environment_step": 1' in received and '"phase_step": 0' in received
    assert '"last_action": {"tool": "move_to"' in received
    assert (
        sum(
            event.get("tool") == "read_text_file" and event["type"] == "tool_call"
            for event in result.messages
        )
        == 1
    )


def test_pass_cannot_override_tool_error_or_native_failure(tmp_path):
    result, scene, _, _ = solve(
        tmp_path,
        [phase("Carry"), phase("Retry")],
        [action(1), check("PASS")],
        scene=Scene(error=True),
        max_turns=3,
        repair_actions=0,
    )
    assert result.finish_result["status"] == "failure"
    assert scene.actions == [1]
    assert "Shared request budget" in result.error
    checked = next(m for m in result.messages if m["type"] == "step_check")
    assert checked["passed"] is False


def test_invalid_verifier_tool_arguments_become_unknown_and_allow_local_repair(
    tmp_path,
):
    scene = Scene(solve_after=2)

    @readonly
    def back_project(row):
        pytest.fail("Invalid tool arguments must not reach the handler")

    scene.add_tool(
        "back_project",
        {
            "name": "back_project",
            "input_schema": {
                "type": "object",
                "properties": {"row": {"type": "integer"}},
                "required": ["row"],
            },
        },
        back_project,
    )
    calls = []

    async def executor(messages, info):
        calls.append(messages)
        if 2 <= len(calls) <= 4:
            return ModelResponse(parts=[ToolCallPart("back_project", {"row": "bad"})])
        return ModelResponse(
            parts=[
                ToolCallPart(info.output_tools[0].name, action(len(scene.actions) + 1))
            ]
        )

    planner = StagedPlanner(
        supervisor_model=scripted([phase("Carry")], []),
        executor_model=FunctionModel(executor),
        output_dir=tmp_path,
        recipe_tag="task",
        dashboard_events=NullDashboardEventSink(),
        repair_actions=1,
    )
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=scene, max_turns=20
    )
    assert result.error is None and result.finish_result["status"] == "success"
    assert scene.actions == [1, 2]
    failed_check = next(
        event for event in result.messages if event["type"] == "step_check"
    )
    assert failed_check["assessment"]["verdict"] == "UNKNOWN"
    assert "max retries" in failed_check["tool_error"]
    assert any(event["type"] == "repair_start" for event in result.messages)
    assert any(
        event["type"] == "validation_retry" and "integer" in event["feedback"]
        for event in result.messages
    )
    assert any(
        event["type"] == "model_tool_call" and event["arguments"] == {"row": "bad"}
        for event in result.messages
    )
    assert '"tool": "move_to"' in str(calls[-1])


def test_model_failure_retains_usage_and_is_not_task_success(tmp_path):
    result, scene, _, _ = solve(
        tmp_path, [phase("Carry")], [RuntimeError("provider unavailable")]
    )
    assert scene.actions == []
    assert "provider unavailable" in result.error
    assert result.finish_result["status"] == "failure"
    assert result.stats["total_input_tokens"] == 100
    assert result.stats["turns_used"] == 2
    assert result.stats["timing"]["model_requests"][1]["status"] == "RuntimeError"
    assert (tmp_path / "task_staged.json").exists()


def test_global_timeout_cancels_and_drains_physical_action(tmp_path):
    scene = Scene(blocking=True)
    result, _, _, _ = solve(
        tmp_path,
        [phase("Carry")],
        [action(1)],
        scene=scene,
        timeout_s=0.1,
    )
    assert scene.started.is_set() and scene.stopped.is_set()
    assert "timed out" in result.error
    assert result.stats["timing"]["tools"][-1]["cancelled"] is True
    assert result.finish_result["status"] == "failure"


def test_global_timeout_also_includes_model_call(tmp_path):
    entered = threading.Event()

    async def slow(messages, info):
        entered.set()
        await asyncio.sleep(5)

    planner = StagedPlanner(
        supervisor_model=FunctionModel(slow),
        executor_model=FunctionModel(slow),
        output_dir=tmp_path,
        recipe_tag="task",
        dashboard_events=NullDashboardEventSink(),
        timeout_s=0.25,
    )
    started = time.perf_counter()
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=Scene(), max_turns=10
    )
    assert "timed out" in result.error
    request = result.stats["timing"]["model_requests"][0]
    assert request["status"] == "CancelledError"
    elapsed = time.perf_counter() - started
    assert entered.is_set()
    # The shared deadline includes Agent construction, not just the model call.
    assert 0 < request["duration_s"] < elapsed
    assert 0.2 <= elapsed < 2


def test_invalid_action_is_corrected_before_robot_execution(tmp_path):
    bad = {"tool": "move_to", "arguments_json": '{"x":"bad"}', "reason": "test"}
    result, scene, _, _ = solve(
        tmp_path, [phase("Carry")], [bad, action(1)], scene=Scene(solve_after=1)
    )
    assert scene.actions == [1]
    assert result.finish_result["status"] == "success"
    assert result.stats["turns_used"] == 3


@pytest.mark.parametrize(
    "supervisor_url,executor_url",
    [
        ("https://supervisor.example/v1", "https://executor.example/v1"),
        ("https://supervisor.example", "https://executor.example/maas"),
    ],
)
def test_credentials_are_independent_and_do_not_mutate_process_env(
    monkeypatch, tmp_path, supervisor_url, executor_url
):
    monkeypatch.setenv("TEST_SUPERVISOR_KEY", "supervisor-key")
    monkeypatch.setenv("TEST_EXECUTOR_KEY", "executor-key")
    monkeypatch.setenv("OPENAI_API_KEY", "original-key")
    planner = build_planner(
        "staged",
        model="openai:gpt-6-astra",
        executor_model="openai-chat:qwen3.6-27b",
        base_url=supervisor_url,
        executor_base_url=executor_url,
        api_key_env="TEST_SUPERVISOR_KEY",
        executor_api_key_env="TEST_EXECUTOR_KEY",
        output_dir=tmp_path,
        recipe_tag="task",
        robot_name="libero",
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner.supervisor_model.client.api_key == "supervisor-key"
    assert planner.executor_model.client.api_key == "executor-key"
    assert str(planner.supervisor_model.client.base_url).startswith(
        "https://supervisor.example"
    )
    assert str(planner.executor_model.client.base_url).startswith(
        "https://executor.example"
    )
    import os

    assert os.environ["OPENAI_API_KEY"] == "original-key"
    assert str(planner.supervisor_model.client.base_url).endswith("/v1/")
    assert str(planner.executor_model.client.base_url).endswith("/v1/")
    assert planner.timeout_s == 1200
    assert build_api_model("openai-chat:test").client.api_key == "original-key"
