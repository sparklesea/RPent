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

"""Helpers for aggregating model usage across internal planner calls."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic_ai.usage import RequestUsage


def merge_request_usage(
    usages: Iterable[RequestUsage | None],
) -> RequestUsage:
    """Return the sum of all non-null request usages."""
    total = RequestUsage()
    for usage in usages:
        if usage is not None:
            total.incr(usage)
    return total


def usage_values(usage: RequestUsage) -> dict[str, int]:
    """Return the token fields recorded for one internal request."""
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": int(usage.cache_read_tokens or 0),
        "cache_write_tokens": int(usage.cache_write_tokens or 0),
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def usage_log_fields(prefix: str, usage: RequestUsage) -> dict[str, int]:
    """Return one request's usage fields with ``prefix`` for a JSONL row."""
    return {f"{prefix}_{key}": value for key, value in usage_values(usage).items()}
