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

"""Request timing at the SDK's actual model-call boundary."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UsageLimitExceeded


def utc_now() -> str:
    """Return an ISO timestamp for correlating remote service logs."""
    return datetime.now(timezone.utc).isoformat()


class RequestAccounting(AbstractCapability):
    """Append timings and reported usage, including failed or cancelled requests.

    Shared lists also let several agents consume one request budget. SDK retries
    count as requests; network retries internal to a provider SDK remain inside
    the recorded duration.
    """

    def __init__(self, records: list[dict], *, role: str, limit: int | None = None):
        self.records = records
        self.role = role
        self.limit = limit

    async def wrap_model_request(self, ctx, *, request_context, handler):
        if self.limit is not None and len(self.records) >= self.limit:
            raise UsageLimitExceeded(f"Shared request budget of {self.limit} reached")
        record = {
            "request": len(self.records) + 1,
            "role": self.role,
            "model": ctx.model.model_name,
            "started_at": utc_now(),
            "status": "pending",
        }
        self.records.append(record)
        started = time.perf_counter()
        try:
            response = await handler(request_context)
            record.update(
                status="ok",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                cache_write_tokens=response.usage.cache_write_tokens,
            )
            return response
        except BaseException as exc:
            record.update(status=type(exc).__name__)
            raise
        finally:
            record.update(
                finished_at=utc_now(),
                duration_s=round(time.perf_counter() - started, 4),
            )
