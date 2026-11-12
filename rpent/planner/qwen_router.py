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

"""Per-turn Qwen routing between a Qwen worker and a senior GPT model.

Enabled by ``RPENT_QWEN_ROUTER=1`` for ``--planner api``. Before every model
request, the existing Qwen worker reviews the unchanged conversation and
returns ``{"route": "qwen" | "gpt", "reason": "..."}``. The selected model then
produces exactly one normal planner response; this experiment does not generate
or select candidates.

The router is called without tool definitions and cannot act on the task. A
missing, malformed, or failed route response falls back to Qwen so a parse
error does not end the episode. The per-task GPT memory named by
``RPENT_QWEN_ROUTER_GPT_MEMORY`` is appended only to GPT requests. The Qwen
branch keeps using the memory already available through the planner's local
memory profile.

GPT environment variables are ``RPENT_QWEN_ROUTER_GPT_MODEL``,
``RPENT_QWEN_ROUTER_GPT_BASE_URL``, ``RPENT_QWEN_ROUTER_GPT_KEY_ENV``,
``RPENT_QWEN_ROUTER_GPT_REASONING_EFFORT`` (default ``high``),
``RPENT_QWEN_ROUTER_GPT_MEMORY``, and
``RPENT_QWEN_ROUTER_TEMPERATURE`` (default ``0``). Every decision is recorded
in ``qwen_router.jsonl`` under the run output directory.
"""

from __future__ import annotations

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

ROUTER_LABEL = "[Qwen router]"
GPT_MEMORY_LABEL = "[GPT task memory]"

ROUTER_INSTRUCTIONS = f"""{ROUTER_LABEL} Decide which planner should produce the next action.

Use the full conversation and the latest observation above. Return "qwen" when the next action is a routine continuation: the current state is clear, the available task memory directly applies, or only a small correction is needed. Return "gpt" when stronger reasoning is likely required: the observation is ambiguous or unexpected, recent actions failed or made no progress, the task needs recovery or a new multi-step strategy, or the next manipulation is contact-sensitive and uncertain.

Do not solve the task, propose an action, or call a tool. Return exactly one JSON object and no other text:
{{"route": "qwen" or "gpt", "reason": "<short reason>"}}"""


def _append_note(messages: list[ModelMessage], note: str) -> list[ModelMessage]:
    """Return a copy of ``messages`` with ``note`` added to the newest request."""
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if isinstance(out[index], ModelRequest):
            out[index] = dataclasses.replace(
                out[index], parts=[*out[index].parts, UserPromptPart(note)]
            )
            return out
    return [*out, ModelRequest(parts=[UserPromptPart(note)])]


def _response_text(response: ModelResponse) -> str:
    """Return all text emitted by a router or planner response."""
    return "\n".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    ).strip()


def _parse_route(text: str) -> tuple[str | None, str]:
    """Parse ``(route, reason)`` from a router response."""
    if not text:
        return None, ""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except ValueError:
        return None, ""
    if not isinstance(value, dict):
        return None, ""
    raw = value.get("route", value.get("model", value.get("decision")))
    reason = str(value.get("reason", value.get("explanation", ""))).strip()
    route = str(raw or "").strip().lower()
    aliases = {
        "qwen": "qwen",
        "local": "qwen",
        "worker": "qwen",
        "gpt": "gpt",
        "senior": "gpt",
        "cloud": "gpt",
    }
    return aliases.get(route), reason


def _object_roots(params: ModelRequestParameters) -> ModelRequestParameters:
    """Restore object-typed roots for strict OpenAI-compatible endpoints."""

    def fix(tool):
        schema = tool.parameters_json_schema
        if schema.get("type") == "object":
            return tool
        return dataclasses.replace(
            tool, parameters_json_schema={**schema, "type": "object"}
        )

    return dataclasses.replace(
        params,
        function_tools=[fix(t) for t in params.function_tools],
        output_tools=[fix(t) for t in params.output_tools],
    )


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


class QwenRouterModel(WrapperModel):
    """Route each request to Qwen or GPT, then generate one response."""

    def __init__(
        self,
        worker: Model,
        senior: Model,
        *,
        router_settings: dict[str, Any] | None = None,
        worker_settings: dict[str, Any] | None = None,
        senior_settings: dict[str, Any] | None = None,
        memory: str = "",
        log_path: Path | None = None,
    ) -> None:
        super().__init__(worker)
        self.senior = senior
        self.router_settings = router_settings or {}
        self.worker_settings = worker_settings or {}
        self.senior_settings = senior_settings or {}
        self.memory = memory.strip()
        self.log_path = log_path
        self.requests = 0
        self.router_calls = 0
        self.route_errors = 0
        self.qwen_routes = 0
        self.gpt_routes = 0
        self.qwen_calls = 0
        self.gpt_calls = 0

    @staticmethod
    def _merge_settings(
        base: ModelSettings | None,
        extra: dict[str, Any],
        *,
        senior: bool = False,
    ) -> ModelSettings:
        settings: dict[str, Any] = dict(base or {})
        if senior:
            # Unified thinking and Qwen provider fields do not apply to GPT.
            settings.pop("thinking", None)
            settings.pop("extra_body", None)
        for key, value in extra.items():
            if key == "extra_body":
                settings["extra_body"] = {
                    **(settings.get("extra_body") or {}),
                    **value,
                }
            else:
                settings[key] = value
        return settings  # type: ignore[return-value]

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    async def _route(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Ask Qwen for a route and return the decision plus its log fields."""
        started = time.monotonic()
        self.router_calls += 1
        try:
            response = await self.wrapped.request(
                _append_note(messages, ROUTER_INSTRUCTIONS),
                self._merge_settings(model_settings, self.router_settings),
                ModelRequestParameters(),
            )
        except Exception as exc:
            self.route_errors += 1
            return (
                "qwen",
                "",
                {
                    "route": "qwen",
                    "route_error": f"{type(exc).__name__}: {exc}"[:2000],
                    "route_fallback": True,
                    "router_latency_s": round(time.monotonic() - started, 3),
                },
            )

        text = _response_text(response)
        route, reason = _parse_route(text)
        record: dict[str, Any] = {
            "route": route or "qwen",
            "route_reason": reason,
            "route_text": text[:2000],
            "router_latency_s": round(time.monotonic() - started, 3),
        }
        if route is None:
            self.route_errors += 1
            record["route_error"] = "invalid router response"
            record["route_fallback"] = True
            return "qwen", reason, record
        return route, reason, record

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests += 1
        started = time.monotonic()
        route, _, record = await self._route(messages, model_settings)
        record = {"request": self.requests, **record}

        if route == "gpt":
            self.gpt_routes += 1
            self.gpt_calls += 1
            note = GPT_MEMORY_LABEL
            if self.memory:
                note += (
                    "\nUse the following task memory only when its stated situation "
                    "matches the current observation:\n" + self.memory
                )
            gpt_started = time.monotonic()
            try:
                response = await self.senior.request(
                    _append_note(messages, note),
                    self._merge_settings(
                        model_settings, self.senior_settings, senior=True
                    ),
                    _object_roots(model_request_parameters),
                )
            except Exception as exc:
                self._log(
                    record
                    | {
                        "selected": "gpt",
                        "gpt_error": f"{type(exc).__name__}: {exc}"[:2000],
                        "gpt_latency_s": round(time.monotonic() - gpt_started, 3),
                        "latency_s": round(time.monotonic() - started, 3),
                    }
                )
                raise
            self._log(
                record
                | {
                    "selected": "gpt",
                    "gpt_latency_s": round(time.monotonic() - gpt_started, 3),
                    "tools": [
                        part.tool_name
                        for part in response.parts
                        if isinstance(part, ToolCallPart)
                    ],
                    "latency_s": round(time.monotonic() - started, 3),
                }
            )
            return response

        self.qwen_routes += 1
        self.qwen_calls += 1
        qwen_started = time.monotonic()
        try:
            response = await self.wrapped.request(
                messages,
                self._merge_settings(model_settings, self.worker_settings),
                model_request_parameters,
            )
        except Exception as exc:
            self._log(
                record
                | {
                    "selected": "qwen",
                    "qwen_error": f"{type(exc).__name__}: {exc}"[:2000],
                    "qwen_latency_s": round(time.monotonic() - qwen_started, 3),
                    "latency_s": round(time.monotonic() - started, 3),
                }
            )
            raise
        self._log(
            record
            | {
                "selected": "qwen",
                "qwen_latency_s": round(time.monotonic() - qwen_started, 3),
                "tools": [
                    part.tool_name
                    for part in response.parts
                    if isinstance(part, ToolCallPart)
                ],
                "latency_s": round(time.monotonic() - started, 3),
            }
        )
        return response

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
    """Read optional task memory used only for GPT-routed requests."""
    if not path or not Path(path).is_file():
        return ""
    return Path(path).read_text(encoding="utf-8")


def qwen_router_from_env(
    worker: Model, *, output_dir: str | Path, reasoning_effort: str
) -> Model:
    """Wrap ``worker`` when ``RPENT_QWEN_ROUTER=1``."""
    if os.environ.get("RPENT_QWEN_ROUTER") != "1":
        return worker
    enabled = [
        name
        for name in ("RPENT_CANDIDATE_SELECTION", "RPENT_ESCALATION")
        if os.environ.get(name) == "1"
    ]
    if enabled:
        raise ValueError(
            "RPENT_QWEN_ROUTER cannot be enabled with " + ", ".join(enabled)
        )

    from rpent.planner.base import build_api_model
    from rpent.planner.prompt_cache import prompt_cache_from_env
    from rpent.planner.retry import retry_from_env

    model_id = os.environ.get("RPENT_QWEN_ROUTER_GPT_MODEL")
    key_env = os.environ.get("RPENT_QWEN_ROUTER_GPT_KEY_ENV")
    if not model_id or not key_env or not os.environ.get(key_env):
        raise ValueError(
            "RPENT_QWEN_ROUTER=1 requires RPENT_QWEN_ROUTER_GPT_MODEL and a "
            "non-empty credential named by RPENT_QWEN_ROUTER_GPT_KEY_ENV"
        )
    senior = retry_from_env(
        prompt_cache_from_env(
            build_api_model(
                model_id,
                os.environ.get("RPENT_QWEN_ROUTER_GPT_BASE_URL") or None,
                api_key=os.environ[key_env],
            )
        )
    )
    worker_settings: dict[str, Any] = {}
    if reasoning_effort == "none" and "qwen" in worker.model_name.lower():
        worker_settings["extra_body"] = {"enable_thinking": False}
    router_settings = {
        **worker_settings,
        "temperature": float(os.environ.get("RPENT_QWEN_ROUTER_TEMPERATURE", "0")),
    }
    senior_settings = {
        "openai_reasoning_effort": os.environ.get(
            "RPENT_QWEN_ROUTER_GPT_REASONING_EFFORT", "high"
        ),
        # Reasoning item ids do not survive turns written by the other model.
        "openai_send_reasoning_ids": False,
    }
    return QwenRouterModel(
        worker,
        senior,
        router_settings=router_settings,
        worker_settings=worker_settings,
        senior_settings=senior_settings,
        memory=_read_memory(os.environ.get("RPENT_QWEN_ROUTER_GPT_MEMORY")),
        log_path=Path(output_dir) / "qwen_router.jsonl",
    )
