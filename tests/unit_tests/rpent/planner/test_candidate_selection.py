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

from rpent.planner.candidate_selection import CandidateSelectionModel


def call(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


class Worker:
    """FunctionModel callback that returns the next scripted candidate."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self.notes = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.notes.append(messages[-1])
        return next(self.replies)


class Selector:
    """FunctionModel callback that records its note and tool definitions."""

    def __init__(self, reply):
        self.reply = reply
        self.notes = []
        self.tool_names = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        self.notes.append(
            " ".join(
                part.content
                for part in last.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        )
        self.tool_names.append([tool.name for tool in info.function_tools])
        return self.reply


def build(worker_replies, selector_reply, **kwargs):
    worker = Worker(worker_replies)
    selector = Selector(selector_reply)
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        **kwargs,
    )
    return model, worker, selector


def ask(model, messages):
    return asyncio.run(model.request(messages, None, ModelRequestParameters()))


START = [ModelRequest.user_text_prompt("start")]


def test_samples_four_candidates_and_returns_verifier_choice():
    model, worker, selector = build(
        [call("left"), call("right"), call("up"), call("down")],
        ModelResponse(
            parts=[TextPart('{"selection": 2, "reason": "right avoids the bowl"}')]
        ),
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "right"
    assert worker.calls == 4
    assert selector.tool_names == [[]]
    assert "Candidate 1" in selector.notes[0]
    assert "Candidate 4" in selector.notes[0]
    assert (model.requests, model.worker_calls, model.selector_calls) == (1, 4, 1)


def test_invalid_selection_falls_back_to_first_candidate():
    model, _, _ = build(
        [call("left"), call("right"), call("up"), call("down")],
        ModelResponse(parts=[TextPart("I cannot decide")]),
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "left"
    assert model.fallbacks == 1


def test_qwen_gate_selects_without_calling_gpt():
    worker = Worker([call("left"), call("right"), call("up"), call("down")])
    gate = Selector(
        ModelResponse(parts=[TextPart('{"selection": 2, "reason": "right is safe"}')])
    )
    selector = Selector(ModelResponse(parts=[TextPart('{"selection": 1}')]))
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        gate_selector=FunctionModel(gate, model_name="gate"),
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "right"
    assert selector.notes == []
    assert (model.gate_calls, model.gate_selections, model.selector_calls) == (1, 1, 0)
    assert "selection 0" in gate.notes[0]


def test_qwen_gate_rejection_escalates_to_gpt():
    worker = Worker([call("left"), call("right"), call("up"), call("down")])
    gate = Selector(
        ModelResponse(parts=[TextPart('{"selection": 0, "reason": "none are safe"}')])
    )
    selector = Selector(ModelResponse(parts=[TextPart('{"selection": 3}')]))
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        gate_selector=FunctionModel(gate, model_name="gate"),
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "up"
    assert model.gate_rejections == 1
    assert model.gpt_escalations == 1
    assert model.selector_calls == 1
    assert "worker gate did not accept" in selector.notes[0]


def test_qwen_gate_invalid_output_escalates_to_gpt():
    worker = Worker([call("left"), call("right")])
    gate = Selector(ModelResponse(parts=[TextPart("I am not sure")]))
    selector = Selector(ModelResponse(parts=[TextPart('{"selection": 2}')]))
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        gate_selector=FunctionModel(gate, model_name="gate"),
        candidate_count=2,
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "right"
    assert model.gate_errors == 1
    assert model.selector_calls == 1


def test_qwen_gate_pair_support_avoids_gpt():
    worker = Worker([call("left"), call("left"), call("up"), call("down")])
    gate = Selector(
        ModelResponse(parts=[TextPart('{"selection": 1, "reason": "pair agrees"}')])
    )
    selector = Selector(ModelResponse(parts=[TextPart('{"selection": 2}')]))
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        gate_selector=FunctionModel(gate, model_name="gate"),
        gate_min_tool_agreement=2,
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "left"
    assert model.gate_selections == 1
    assert model.gate_disagreements == 0
    assert model.selector_calls == 0


def test_qwen_gate_isolated_choice_escalates_to_gpt():
    worker = Worker([call("left"), call("right"), call("up"), call("down")])
    gate = Selector(
        ModelResponse(parts=[TextPart('{"selection": 1, "reason": "left is best"}')])
    )
    selector = Selector(ModelResponse(parts=[TextPart('{"selection": 2}')]))
    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(selector, model_name="selector"),
        gate_selector=FunctionModel(gate, model_name="gate"),
        gate_min_tool_agreement=2,
    )

    response = ask(model, START)

    assert response.parts[0].tool_name == "right"
    assert model.gate_selections == 0
    assert model.gate_disagreements == 1
    assert model.gpt_escalations == 1
    assert model.selector_calls == 1
    assert "worker gate did not accept" in selector.notes[0]


def test_text_only_candidate_is_not_sent_to_selector():
    model, _, selector = build(
        [ModelResponse(parts=[TextPart("done")])],
        ModelResponse(parts=[TextPart('{"selection": 1}')]),
        candidate_count=1,
    )

    response = ask(model, START)

    assert response.parts[0].content == "done"
    assert selector.notes == []
    assert model.selector_calls == 0
    assert model.fallbacks == 1


def test_memory_reaches_selector_but_not_worker_or_original_messages():
    model, worker, selector = build(
        [call("left"), call("right")],
        ModelResponse(parts=[TextPart('{"selection": 1}')]),
        candidate_count=2,
        memory="Lift the mug before moving sideways.",
    )

    ask(model, START)

    assert "Lift the mug" in selector.notes[0]
    assert not any(
        isinstance(part, UserPromptPart) and "Lift the mug" in str(part.content)
        for part in START[0].parts
    )
    assert worker.calls == 2


def test_selection_parser_accepts_numbered_or_fenced_text():
    from rpent.planner.candidate_selection import _parse_selection

    assert _parse_selection("Candidate 3 is best", 4) == (3, "")
    assert _parse_selection('```json\n{"selection": 2}\n```', 4) == (2, "")
    assert _parse_selection('{"selection": 9}', 4) == (None, "")


def test_gate_parser_distinguishes_rejection_from_invalid_output():
    from rpent.planner.candidate_selection import _parse_gate_selection

    assert _parse_gate_selection('{"selection": 0}', 4) == (None, True, "")
    assert _parse_gate_selection('{"selection": 2}', 4) == (2, False, "")
    assert _parse_gate_selection("none are acceptable", 4) == (None, True, "")
    assert _parse_gate_selection("I cannot decide", 4) == (None, False, "")


def test_streamed_agent_run_uses_selected_candidate():
    candidate_round = 0

    def worker(messages, info: AgentInfo) -> ModelResponse:
        nonlocal candidate_round
        candidate_round += 1
        if candidate_round <= 4:
            return call("noop", candidate=candidate_round)
        return ModelResponse(parts=[TextPart("done")])

    model = CandidateSelectionModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart('{"selection": 1}')]),
            model_name="selector",
        ),
        candidate_count=2,
    )
    agent = Agent(model)

    @agent.tool_plain
    def noop(candidate: int) -> str:
        return f"candidate {candidate}"

    async def handler(ctx, events):
        async for _ in events:
            pass

    output = agent.run_sync("start", event_stream_handler=handler)

    assert output.output == "done"
    assert model.requests >= 2


def test_candidate_selection_and_escalation_are_mutually_exclusive(monkeypatch):
    from rpent.planner.candidate_selection import candidate_selection_from_env

    monkeypatch.setenv("RPENT_CANDIDATE_SELECTION", "1")
    monkeypatch.setenv("RPENT_ESCALATION", "1")

    try:
        candidate_selection_from_env(
            FunctionModel(lambda messages, info: ModelResponse(parts=[])),
            output_dir=".",
            reasoning_effort="none",
        )
    except ValueError as error:
        assert "cannot be enabled together" in str(error)
    else:
        raise AssertionError("expected mutually exclusive configuration error")
