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

"""Best-of-M candidate generation with a senior verifier (opt-in).

Enabled by ``RPENT_CANDIDATE_SELECTION=1`` for ``--planner api``. For every
model request, the worker sampled from ``--model`` generates ``M`` complete
candidate responses. A selector model then receives the unchanged
conversation plus a compact rendering of those candidates and returns the
one-based index of the response to execute. The original ``messages`` are
never modified, and only the selected response is returned to the agent loop.

The selector is deliberately called without tool definitions: it chooses
among the worker's proposals instead of inventing another action. Invalid or
failed selections fall back to the first valid candidate so an evaluation
episode is not lost to a verifier parsing error; every fallback is recorded in
``candidate_selection.jsonl`` under the run output directory.

With ``RPENT_CANDIDATE_QWEN_GATE=1``, the worker model first reviews its own
candidates. It returns one candidate when at least one is acceptable, or
``selection: 0`` when none are. Only the latter case calls the senior selector,
so GPT is reserved for turns where the worker rejects its complete candidate
set. ``RPENT_CANDIDATE_GATE_TEMPERATURE`` controls gate determinism (default
0). ``RPENT_CANDIDATE_GATE_MIN_TOOL_AGREEMENT`` can additionally defer a gate
selection to the senior selector unless the selected ordered tool sequence has
enough matching candidates. The gate and selector both receive the unchanged
conversation; only the candidate rendering and their decision prompts differ.

Worker sampling can run concurrently with ``RPENT_CANDIDATE_CONCURRENCY``
(default 1). ``RPENT_CANDIDATE_TEMPERATURE`` and ``RPENT_CANDIDATE_TOP_P``
optionally control sampling diversity. Selector environment variables are
``RPENT_CANDIDATE_SELECTOR_MODEL``, ``RPENT_CANDIDATE_SELECTOR_BASE_URL``,
``RPENT_CANDIDATE_SELECTOR_KEY_ENV`` and
``RPENT_CANDIDATE_SELECTOR_REASONING_EFFORT``. The selector never reads
``RPENT_CLOUD_CORRECTIONS``; task memory can instead be supplied through
``RPENT_CANDIDATE_SELECTOR_MEMORY``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from rpent.planner.usage import merge_request_usage, usage_log_fields, usage_values

SELECTOR_LABEL = "[candidate verifier]"


def _has_tool_call(response: ModelResponse) -> bool:
    """Return whether ``response`` proposes at least one tool call."""
    return any(isinstance(part, ToolCallPart) for part in response.parts)


def _append_note(messages: list[ModelMessage], note: str) -> list[ModelMessage]:
    """Return a copy of ``messages`` with ``note`` appended to its final request."""
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if isinstance(out[index], ModelRequest):
            out[index] = dataclasses.replace(
                out[index], parts=[*out[index].parts, UserPromptPart(note)]
            )
            return out
    return [*out, ModelRequest(parts=[UserPromptPart(note)])]


def _candidate_record(response: ModelResponse) -> dict[str, Any]:
    """Serialize the parts a verifier needs from one candidate response."""
    texts = [part.content for part in response.parts if isinstance(part, TextPart)]
    calls: list[dict[str, Any]] = []
    for part in response.parts:
        if not isinstance(part, ToolCallPart):
            continue
        try:
            args = part.args_as_dict()
        except ValueError:
            args = part.args
        calls.append({"tool": part.tool_name, "args": args})
    return {"text": "\n".join(texts), "tool_calls": calls}


def _tool_sequence(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Return the ordered tool names proposed by one candidate."""
    return tuple(
        str(call.get("tool"))
        for call in candidate.get("tool_calls") or []
        if call.get("tool")
    )


def _tool_agreement(candidates: list[dict[str, Any]], index: int) -> int:
    """Count candidates whose ordered tool sequence matches ``index``."""
    sequence = _tool_sequence(candidates[index])
    return sum(_tool_sequence(candidate) == sequence for candidate in candidates)


def _render_candidates(
    candidates: list[dict[str, Any]],
    memory: str = "",
    *,
    allow_none: bool = False,
    intro: str | None = None,
) -> str:
    """Build the verifier note from compact candidate records."""
    lines = [intro or "[harness] Choose exactly one candidate action to execute next."]
    lines.extend(
        [
            "The latest observation is the final tool result(s) and images in the "
            "conversation above. Compare every candidate against the task, the "
            "current robot state, and the remaining step budget.",
        ]
    )
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"Candidate {index}:")
        text = str(candidate.get("text") or "").strip()
        if text:
            lines.append(f"  Rationale: {text[:1200]}")
        calls = candidate.get("tool_calls") or []
        if not calls:
            lines.append("  Tool calls: none")
        for call in calls:
            lines.append(
                f"  Tool call: {call.get('tool')} "
                f"args={json.dumps(call.get('args'), ensure_ascii=False, default=str)}"
            )
    if memory:
        lines.append(
            "Task memory for judging candidate suitability (apply an entry only "
            "when its situation matches the current observation):\n" + memory
        )
    lines.append("Do not propose a new action and do not call tools.")
    if allow_none:
        lines.extend(
            [
                "Choose a candidate only if it is safe and useful now. If none are "
                "acceptable, return selection 0. Return JSON only:",
                '{"selection": <one-based candidate number, or 0 if none>, '
                '"reason": "<short reason>"}',
            ]
        )
    else:
        lines.extend(
            [
                "Return JSON only:",
                '{"selection": <one-based candidate number>, '
                '"reason": "<short reason>"}',
            ]
        )
    return "\n".join(lines)


def _response_text(response: ModelResponse) -> str:
    """Return all text emitted by a verifier response."""
    return "\n".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    ).strip()


def _parse_selection(text: str, count: int) -> tuple[int | None, str]:
    """Parse a one-based verifier selection, returning ``(index, reason)``."""
    if not text:
        return None, ""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except ValueError:
        value = None
    if isinstance(value, dict):
        raw = value.get("selection", value.get("choice", value.get("candidate")))
        reason = str(value.get("reason", value.get("explanation", ""))).strip()
        try:
            selection = int(raw)
        except (TypeError, ValueError):
            selection = None
        if selection is not None and 1 <= selection <= count:
            return selection, reason

    match = re.search(r"(?i)\b(?:selection|choice|candidate)\s*[:=#-]?\s*(\d+)\b", text)
    if match is None:
        match = re.search(r"(?<![-\w])(\d+)(?![-\w])", text)
    if match is None:
        return None, ""
    selection = int(match.group(1))
    return (selection, "") if 1 <= selection <= count else (None, "")


def _parse_gate_selection(text: str, count: int) -> tuple[int | None, bool, str]:
    """Parse a gate decision and distinguish a rejection from malformed output."""
    if not text:
        return None, False, ""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except ValueError:
        value = None
    if isinstance(value, dict):
        raw = value.get("selection", value.get("choice", value.get("candidate")))
        reason = str(value.get("reason", value.get("explanation", ""))).strip()
        if raw is None or str(raw).strip().lower() in {
            "0",
            "none",
            "no",
            "reject",
            "reject_all",
        }:
            return None, True, reason
        try:
            selection = int(raw)
        except (TypeError, ValueError):
            selection = None
        if selection is not None and 1 <= selection <= count:
            return selection, False, reason
    if re.search(r"(?i)\b(?:none|no acceptable|reject all)\b", text):
        return None, True, ""
    selection, reason = _parse_selection(text, count)
    return selection, False, reason


@dataclasses.dataclass
class _ReplayStream(StreamedResponse):
    """Present an already completed response through the streaming interface."""

    _response: ModelResponse = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        self._usage = self._response.usage
        self.provider_response_id = self._response.provider_response_id
        self.provider_details = self._response.provider_details
        self.finish_reason = self._response.finish_reason

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._response.parts):
            yield self._parts_manager.handle_part(vendor_part_id=index, part=part)

    async def close_stream(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self._response.model_name or ""

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp


class CandidateSelectionModel(WrapperModel):
    """Sample several worker responses and return the verifier's choice."""

    def __init__(
        self,
        worker: Model,
        selector: Model,
        *,
        gate_selector: Model | None = None,
        candidate_count: int = 4,
        concurrency: int = 1,
        gate_min_tool_agreement: int = 1,
        worker_settings: dict[str, Any] | None = None,
        gate_settings: dict[str, Any] | None = None,
        selector_settings: dict[str, Any] | None = None,
        memory: str = "",
        log_path: Path | None = None,
    ) -> None:
        super().__init__(worker)
        if candidate_count < 1 or concurrency < 1:
            raise ValueError("candidate_count and concurrency must be positive")
        if not 1 <= gate_min_tool_agreement <= candidate_count:
            raise ValueError(
                "gate_min_tool_agreement must be between 1 and candidate_count"
            )
        self.selector = selector
        self.gate_selector = gate_selector
        self.candidate_count = candidate_count
        self.concurrency = min(concurrency, candidate_count)
        self.gate_min_tool_agreement = gate_min_tool_agreement
        self.worker_settings = worker_settings or {}
        self.gate_settings = gate_settings or {}
        self.selector_settings = selector_settings or {}
        self.memory = memory.strip()
        self.log_path = log_path
        self.requests = 0
        self.worker_calls = 0
        self.gate_calls = 0
        self.gate_selections = 0
        self.gate_rejections = 0
        self.gate_disagreements = 0
        self.gate_errors = 0
        self.selector_calls = 0
        self.gpt_escalations = 0
        self.fallbacks = 0

    @staticmethod
    def _merge_settings(
        base: ModelSettings | None,
        extra: dict[str, Any],
        *,
        selector: bool,
    ) -> ModelSettings:
        settings: dict[str, Any] = dict(base or {})
        if selector:
            settings.pop("thinking", None)
            settings.pop("extra_body", None)
            settings.pop("tool_choice", None)
            settings.pop("parallel_tool_calls", None)
        for key, value in extra.items():
            if key == "extra_body":
                settings["extra_body"] = {
                    **(settings.get("extra_body") or {}),
                    **value,
                }
            else:
                settings[key] = value
        return settings  # type: ignore[return-value]

    async def _generate_candidates(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        params: ModelRequestParameters,
    ) -> list[ModelResponse]:
        settings = self._merge_settings(
            model_settings, self.worker_settings, selector=False
        )

        async def sample() -> ModelResponse:
            return await self.wrapped.request(messages, settings, params)

        if self.concurrency == 1:
            return [await sample() for _ in range(self.candidate_count)]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def limited_sample() -> ModelResponse:
            async with semaphore:
                return await sample()

        return list(
            await asyncio.gather(
                *(limited_sample() for _ in range(self.candidate_count))
            )
        )

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests += 1
        started = time.monotonic()
        try:
            responses = await self._generate_candidates(
                messages, model_settings, model_request_parameters
            )
        except Exception as exc:
            self._log(
                {
                    "request": self.requests,
                    "candidate_count": self.candidate_count,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                    "latency_s": round(time.monotonic() - started, 3),
                }
            )
            raise
        self.worker_calls += len(responses)
        worker_usage = merge_request_usage(response.usage for response in responses)
        records = [_candidate_record(response) for response in responses]
        valid = [index for index, record in enumerate(records) if record["tool_calls"]]
        generated_at = round(time.monotonic() - started, 3)
        row: dict[str, Any] = {
            "request": self.requests,
            "candidate_count": self.candidate_count,
            "valid_count": len(valid),
            "candidates": records,
            "worker_latency_s": generated_at,
            "candidate_usage": [usage_values(response.usage) for response in responses],
            **usage_log_fields("worker", worker_usage),
        }
        if not valid:
            self.fallbacks += 1
            self._log(
                row
                | usage_log_fields("total", worker_usage)
                | {
                    "selected": 1,
                    "selected_candidate": 1,
                    "invalid_selection": True,
                    "selection_reason": "no candidate emitted a tool call",
                    "latency_s": generated_at,
                }
            )
            return dataclasses.replace(responses[0], usage=worker_usage)

        selected_records = [records[index] for index in valid]
        gate_record: dict[str, Any] = {}
        gate_usage: RequestUsage | None = None
        if self.gate_selector is not None:
            gate_note = _render_candidates(selected_records, allow_none=True)
            gate_started = time.monotonic()
            try:
                gate_choice = await self.gate_selector.request(
                    _append_note(messages, gate_note),
                    self._merge_settings(
                        model_settings, self.gate_settings, selector=False
                    ),
                    ModelRequestParameters(),
                )
                gate_usage = gate_choice.usage
                self.gate_calls += 1
                gate_selection, gate_none, gate_reason = _parse_gate_selection(
                    _response_text(gate_choice), len(valid)
                )
                gate_record = {
                    "gate_text": _response_text(gate_choice)[:2000],
                    "gate_latency_s": round(time.monotonic() - gate_started, 3),
                    "gate_selection": gate_selection,
                    "gate_reason": gate_reason,
                    **usage_log_fields("gate", gate_usage),
                }
                if gate_selection is not None:
                    tool_agreement = _tool_agreement(
                        selected_records, gate_selection - 1
                    )
                    gate_record.update(
                        {
                            "selected_tool_agreement": tool_agreement,
                            "gate_min_tool_agreement": self.gate_min_tool_agreement,
                        }
                    )
                    if tool_agreement >= self.gate_min_tool_agreement:
                        self.gate_selections += 1
                        total_usage = merge_request_usage([worker_usage, gate_usage])
                        self._log(
                            row
                            | gate_record
                            | usage_log_fields("total", total_usage)
                            | {
                                "gate_decision": "select",
                                "selected": gate_selection,
                                "selected_candidate": valid[gate_selection - 1] + 1,
                                "invalid_selection": False,
                                "selection_reason": gate_reason,
                                "latency_s": round(time.monotonic() - started, 3),
                            }
                        )
                        return dataclasses.replace(
                            responses[valid[gate_selection - 1]],
                            usage=total_usage,
                        )
                    self.gate_disagreements += 1
                    gate_record["gate_decision"] = "disagreement"
                elif gate_none:
                    self.gate_rejections += 1
                    gate_record["gate_decision"] = "none"
                else:
                    self.gate_errors += 1
                    gate_record["gate_decision"] = "invalid"
            except Exception as exc:
                self.gate_errors += 1
                gate_record = {
                    "gate_decision": "error",
                    "gate_error": f"{type(exc).__name__}: {exc}"[:2000],
                    "gate_latency_s": round(time.monotonic() - gate_started, 3),
                    **usage_log_fields("gate", RequestUsage()),
                }

        intro = None
        if gate_record:
            intro = (
                "[harness] The worker gate did not accept any candidate. "
                "Choose exactly one candidate action to execute next."
            )
        note = _render_candidates(selected_records, self.memory, intro=intro)
        selector_started = time.monotonic()
        self.gpt_escalations += 1
        try:
            choice = await self.selector.request(
                _append_note(messages, note),
                self._merge_settings(
                    model_settings, self.selector_settings, selector=True
                ),
                ModelRequestParameters(),
            )
            self.selector_calls += 1
        except Exception as exc:
            self.fallbacks += 1
            total_usage = merge_request_usage([worker_usage, gate_usage])
            self._log(
                row
                | gate_record
                | usage_log_fields("total", total_usage)
                | {
                    "gpt_escalated": bool(gate_record),
                    "selected": 1,
                    "selected_candidate": valid[0] + 1,
                    "invalid_selection": True,
                    "selection_error": f"{type(exc).__name__}: {exc}"[:2000],
                    "selector_latency_s": round(time.monotonic() - selector_started, 3),
                    "latency_s": round(time.monotonic() - started, 3),
                }
            )
            return dataclasses.replace(
                responses[valid[0]],
                usage=total_usage,
            )

        selection, reason = _parse_selection(_response_text(choice), len(valid))
        invalid = selection is None
        if selection is None:
            selection = 1
            self.fallbacks += 1
        selected_response = responses[valid[selection - 1]]
        total_usage = merge_request_usage([worker_usage, gate_usage, choice.usage])
        self._log(
            row
            | gate_record
            | usage_log_fields("selector", choice.usage)
            | usage_log_fields("total", total_usage)
            | {
                "gpt_escalated": bool(gate_record),
                "selected": selection,
                "selected_candidate": valid[selection - 1] + 1,
                "invalid_selection": invalid,
                "selection_reason": reason,
                "selector_text": _response_text(choice)[:2000],
                "selector_latency_s": round(time.monotonic() - selector_started, 3),
                "latency_s": round(time.monotonic() - started, 3),
            }
        )
        return dataclasses.replace(selected_response, usage=total_usage)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        response = await self.request(
            messages, model_settings, model_request_parameters
        )
        yield _ReplayStream(
            model_request_parameters=model_request_parameters, _response=response
        )


def _read_memory(path: str | None) -> str:
    """Read optional task memory used only in the verifier note."""
    if not path or not Path(path).is_file():
        return ""
    return Path(path).read_text(encoding="utf-8")


def candidate_selection_from_env(
    worker: Model, *, output_dir: str | Path, reasoning_effort: str
) -> Model:
    """Wrap ``worker`` when ``RPENT_CANDIDATE_SELECTION=1``."""
    if os.environ.get("RPENT_CANDIDATE_SELECTION") != "1":
        return worker
    if os.environ.get("RPENT_ESCALATION") == "1":
        raise ValueError(
            "RPENT_CANDIDATE_SELECTION and RPENT_ESCALATION cannot be enabled together"
        )
    from rpent.planner.base import build_api_model
    from rpent.planner.prompt_cache import prompt_cache_from_env
    from rpent.planner.retry import retry_from_env

    model_id = os.environ.get("RPENT_CANDIDATE_SELECTOR_MODEL")
    key_env = os.environ.get("RPENT_CANDIDATE_SELECTOR_KEY_ENV")
    if not model_id or not key_env or not os.environ.get(key_env):
        raise ValueError(
            "RPENT_CANDIDATE_SELECTION=1 requires RPENT_CANDIDATE_SELECTOR_MODEL "
            "and a non-empty credential named by "
            "RPENT_CANDIDATE_SELECTOR_KEY_ENV"
        )
    selector = retry_from_env(
        prompt_cache_from_env(
            build_api_model(
                model_id,
                os.environ.get("RPENT_CANDIDATE_SELECTOR_BASE_URL") or None,
                api_key=os.environ[key_env],
            )
        )
    )
    worker_settings: dict[str, Any] = {}
    if reasoning_effort == "none" and "qwen" in worker.model_name.lower():
        worker_settings["extra_body"] = {"enable_thinking": False}
    if value := os.environ.get("RPENT_CANDIDATE_TEMPERATURE"):
        worker_settings["temperature"] = float(value)
    if value := os.environ.get("RPENT_CANDIDATE_TOP_P"):
        worker_settings["top_p"] = float(value)
    selector_settings = {
        "openai_reasoning_effort": os.environ.get(
            "RPENT_CANDIDATE_SELECTOR_REASONING_EFFORT", "high"
        ),
        "openai_send_reasoning_ids": False,
    }
    qwen_gate = os.environ.get("RPENT_CANDIDATE_QWEN_GATE") == "1"
    gate_min_tool_agreement = int(
        os.environ.get("RPENT_CANDIDATE_GATE_MIN_TOOL_AGREEMENT", "1")
    )
    gate_settings: dict[str, Any] = {}
    if qwen_gate:
        gate_settings = {
            "extra_body": {"enable_thinking": False},
            "temperature": float(
                os.environ.get("RPENT_CANDIDATE_GATE_TEMPERATURE", "0")
            ),
        }
    return CandidateSelectionModel(
        worker,
        selector,
        gate_selector=worker if qwen_gate else None,
        candidate_count=int(os.environ.get("RPENT_CANDIDATE_COUNT", "4")),
        concurrency=int(os.environ.get("RPENT_CANDIDATE_CONCURRENCY", "1")),
        gate_min_tool_agreement=gate_min_tool_agreement,
        worker_settings=worker_settings,
        gate_settings=gate_settings,
        selector_settings=selector_settings,
        memory=_read_memory(os.environ.get("RPENT_CANDIDATE_SELECTOR_MEMORY")),
        log_path=Path(output_dir) / "candidate_selection.jsonl",
    )
