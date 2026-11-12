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

"""Toolkit overrides for compact RoboTwin evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robots.robotwin.toolkit import RoboTwinToolkit
from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root

_GUIDE_PATH = (
    Path(get_repo_root()) / "robots" / "robotwin" / "guides" / "GUIDE_RPENT.md"
)

_COMPACT_LINGBOT_SPEC: dict[str, Any] = {
    "name": "lingbot_act",
    "description": (
        "Run LingBot-VLA eef16 actions using the native task instruction. "
        "The runtime supplies the task text and use_length=50; provide only chunks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunks": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["chunks"],
    },
}

_GUIDE_REPLACEMENTS = (
    (
        "- Resources: `list_dir`, `read_text_file`",
        "- Resources: `read_memory` (pass a short ref; do not use filesystem paths)",
    ),
    (
        "Every `lingbot_act` must use the full current task language and `use_length=50`.",
        "`lingbot_act` requires only `chunks` (1-3). The runtime applies the "
        "complete current task_language and use_length=50.",
    ),
    (
        "after the primitive and hand back with the same complete task language, "
        "normally\nusing one chunk near contact.",
        "after the primitive and continue with `lingbot_act`; the runtime applies "
        "the complete current task_language. Normally use one chunk near contact.",
    ),
)


class CompactRoboTwinToolkit(RoboTwinToolkit):
    """RoboTwin toolkit that exposes memory refs and compact VLA arguments."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: Any,
        memory: Any,
        reference_tag: str,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
    ) -> None:
        self._reference_tag = reference_tag
        super().__init__(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=dashboard_events,
            memory=memory,
            mode=mode,
            attempts_per_session=attempts_per_session,
            state_output_dir=state_output_dir,
        )

    def _register_robotwin_tools(self) -> None:
        super()._register_robotwin_tools()
        self._tools.pop("read_text_file", None)
        self._tools.pop("list_dir", None)
        self.add_tool(
            "lingbot_act",
            _COMPACT_LINGBOT_SPEC,
            self._compact_lingbot_act,
        )
        refs = self._memory_refs()
        self.add_tool(
            "read_memory",
            {
                "name": "read_memory",
                "description": (
                    "Read one approved RoboTwin guide or memory document by a "
                    "short ref. No filesystem path is needed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "enum": sorted(refs),
                        }
                    },
                    "required": ["ref"],
                },
            },
            self._read_memory,
        )

    def _memory_refs(self) -> dict[str, Path]:
        root = self.memory.root
        refs = {
            "guide": _GUIDE_PATH,
            "task_info": root / "task-specific" / f"{self._reference_tag}.json",
            "task_recipe": root
            / "task-specific"
            / f"{self._reference_tag}_recipe.jsonl",
            "memory_index": root / "MEMORY.md",
        }
        global_dir = root / "global"
        if global_dir.is_dir():
            for path in sorted(global_dir.glob("*.md")):
                refs[path.stem] = path
        return refs

    @readonly
    def _read_memory(self, *, ref: str) -> dict[str, Any]:
        normalized = ref.strip()
        if normalized.startswith("global/"):
            normalized = normalized[len("global/") :]
        if normalized.endswith(".md"):
            normalized = normalized[:-3]
        refs = self._memory_refs()
        path = refs.get(normalized)
        if path is None:
            return {
                "error": f"unknown memory ref: {ref}",
                "available_refs": sorted(refs),
            }
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            return {
                "error": f"memory ref unavailable: {normalized}",
                "detail": str(error),
            }
        if normalized == "guide":
            for old, new in _GUIDE_REPLACEMENTS:
                content = content.replace(old, new)
        if len(content) > 40000:
            content = content[:40000] + "\n\n[TRUNCATED at 40000 chars]"
        return {"ref": normalized, "content": content}

    def _compact_lingbot_act(self, *, chunks: int) -> dict[str, Any]:
        result = self._step("lingbot_act", chunks=chunks)
        if isinstance(result, dict):
            result.pop("agent_prompt_ignored", None)
            result.pop("ignored_agent_prompt", None)
            result.pop("prompt", None)
        return result
