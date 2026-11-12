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

import asyncio
import json

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rpent.planner.qwen_router import QwenRouterModel


def call(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


def route(choice, reason="test route"):
    return ModelResponse(
        parts=[TextPart(json.dumps({"route": choice, "reason": reason}))]
    )


class Qwen:
    """Return a router decision for the router request, then one Qwen action."""

    def __init__(self, route_reply, action_reply=None):
        self.route_reply = route_reply
        self.action_reply = action_reply or call("qwen_action")
        self.action_index = 0
        self.calls = 0
        self.notes = []
        self.tool_names = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        self.calls += 1
        last = messages[-1]
        note = " ".join(
            str(part.content) for part in last.parts if isinstance(part, UserPromptPart)
        )
        self.notes.append(note)
        self.tool_names.append([tool.name for tool in info.function_tools])
        if "[Qwen router]" in note:
            if isinstance(self.route_reply, Exception):
                raise self.route_reply
            return self.route_reply
        if isinstance(self.action_reply, list):
            reply = self.action_reply[self.action_index]
            self.action_index += 1
            return reply
        return self.action_reply


class Senior:
    """Record the GPT request and return one senior action."""

    def __init__(self, reply=None):
        self.reply = reply or call("gpt_action")
        self.calls = 0
        self.notes = []
        self.tool_names = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        self.calls += 1
        last = messages[-1]
        self.notes.append(
            " ".join(
                str(part.content)
                for part in last.parts
                if isinstance(part, UserPromptPart)
            )
        )
        self.tool_names.append([tool.name for tool in info.function_tools])
        return self.reply


def build(route_reply, *, action_reply=None, memory="", log_path=None):
    qwen = Qwen(route_reply, action_reply)
    senior = Senior()
    model = QwenRouterModel(
        FunctionModel(qwen, model_name="qwen"),
        FunctionModel(senior, model_name="gpt"),
        memory=memory,
        log_path=log_path,
    )
    return model, qwen, senior


def ask(model, messages):
    return asyncio.run(model.request(messages, None, ModelRequestParameters()))


START = [ModelRequest.user_text_prompt("start")]


def test_qwen_route_calls_router_then_qwen_once():
    model, qwen, senior = build(route("qwen"))

    response = ask(model, START)

    assert response.parts[0].tool_name == "qwen_action"
    assert qwen.calls == 2
    assert qwen.tool_names == [[], []]
    assert senior.calls == 0
    assert (model.router_calls, model.qwen_routes, model.gpt_routes) == (1, 1, 0)
    assert (model.qwen_calls, model.gpt_calls) == (1, 0)


def test_gpt_route_calls_router_then_senior_with_memory_only():
    model, qwen, senior = build(
        route("gpt", "the scene needs recovery"),
        memory="Inspect the drawer before grasping.",
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "gpt_action"
    assert qwen.calls == 1
    assert senior.calls == 1
    assert "Inspect the drawer" not in qwen.notes[0]
    assert "Inspect the drawer" in senior.notes[0]
    assert "the scene needs recovery" not in senior.notes[0]
    assert (model.qwen_routes, model.gpt_routes) == (0, 1)
    assert (model.qwen_calls, model.gpt_calls) == (0, 1)


def test_invalid_route_falls_back_to_qwen_and_is_logged(tmp_path):
    log_path = tmp_path / "qwen_router.jsonl"
    model, qwen, senior = build(
        ModelResponse(parts=[TextPart("I think GPT should handle this")]),
        log_path=log_path,
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "qwen_action"
    assert qwen.calls == 2
    assert senior.calls == 0
    assert model.route_errors == 1
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["route"] == "qwen"
    assert row["route_fallback"] is True
    assert row["route_error"] == "invalid router response"
    assert row["selected"] == "qwen"


def test_router_exception_falls_back_to_qwen(tmp_path):
    log_path = tmp_path / "qwen_router.jsonl"
    model, qwen, senior = build(RuntimeError("router unavailable"), log_path=log_path)

    response = ask(model, START)

    assert response.parts[0].tool_name == "qwen_action"
    assert qwen.calls == 2
    assert senior.calls == 0
    assert model.route_errors == 1
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert "router unavailable" in row["route_error"]
    assert row["route_fallback"] is True


def test_original_messages_are_not_modified():
    model, _, _ = build(route("gpt"), memory="Use the handle.")

    ask(model, START)

    assert len(START[0].parts) == 1
    assert START[0].parts[0].content == "start"


def test_route_parser_accepts_aliases_and_fenced_json():
    from rpent.planner.qwen_router import _parse_route

    assert _parse_route('{"route": "qwen", "reason": "clear"}') == ("qwen", "clear")
    assert _parse_route('```json\n{"decision": "senior"}\n```') == ("gpt", "")
    assert _parse_route('{"model": "local"}') == ("qwen", "")
    assert _parse_route("plain text") == (None, "")


def test_streamed_agent_run_uses_routed_model():
    model, qwen, _ = build(
        route("qwen"),
        action_reply=[
            call("qwen_action"),
            ModelResponse(parts=[TextPart("continued")]),
        ],
    )
    agent = Agent(model)

    @agent.tool_plain
    def qwen_action() -> str:
        return "continued"

    async def handler(ctx, events):
        async for _ in events:
            pass

    output = agent.run_sync("start", event_stream_handler=handler)

    assert output.output == "continued"
    assert model.requests == 2
    assert qwen.calls == 4


def test_router_mode_is_mutually_exclusive(monkeypatch):
    from rpent.planner.qwen_router import qwen_router_from_env

    monkeypatch.setenv("RPENT_QWEN_ROUTER", "1")
    monkeypatch.setenv("RPENT_ESCALATION", "1")

    try:
        qwen_router_from_env(
            FunctionModel(lambda messages, info: ModelResponse(parts=[])),
            output_dir=".",
            reasoning_effort="none",
        )
    except ValueError as error:
        assert "cannot be enabled with" in str(error)
    else:
        raise AssertionError("expected mutually exclusive configuration error")
