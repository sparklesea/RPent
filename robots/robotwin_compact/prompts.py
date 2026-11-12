"""Prompts for the compact RoboTwin evaluation variant."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robotwin.prompts import evaluate as base_evaluate
from robots.robotwin.prompts import user as base_user
from rpent.prompt.utils import PromptNode

READ_ORDER = """Before the first robot mutation:
1. Read read_memory(ref="guide") completely.
2. Inspect view_env_state(step=0) and its head image.
3. Read read_memory(ref="task_info") and read_memory(ref="task_recipe") when present.
4. Read read_memory(ref="memory_index") and at most one to three relevant leaves by
   their short ref, for example read_memory(ref="grasp_and_hold").

The current task_language and fresh observation override historical memory.
Use the semantic JSON as the phase plan and the JSONL as evidence for action
type and VLA cadence, never as a coordinate replay."""

CONTROL = """lingbot_act takes only chunks; the runtime applies the complete current
task_language and use_length=50. Use one chunk near contact, near success,
instability, or for a small correction; two for ordinary stable progress; three
only for a recipe-supported continuity-sensitive phase already moving correctly.
When VLA has correct contact and visible progress, avoid interrupting it with
speculative primitives. Repeated VLA calls are allowed; after an unproductive
chunk, use fresh evidence to choose whether to continue, shorten the next chunk,
or improve binding, visibility, or physical staging first.

Prefer VLA for grasp/re-grasp, receiving-arm grasp, bimanual coordination,
insertion, hanging, tool use, and contact-rich motion. Use primitives after
verified state for measured free-space transport, staging, retreat, release, or
one small geometric correction. Never transport because a gripper merely looks
closed: also require visible target motion, elevation, or an emptied source.
Never call a primitive just to test whether it helps. For planner residuals,
guarded low approaches, physical state shaping, and wrist-sweep safety, follow
the guide and re-observe after every primitive."""

RUNTIME = """The registered RoboTwin Toolkit is the only control surface. Do
not use shell, Python, network clients, legacy command files, plan mode, user
questions, or unrelated built-in tools. Never inspect task source, evaluator
implementation, hidden rewards, object poses, raw expert trajectories, another
attempt, or unapproved historical geometry. Files exposed through read_memory
are approved planning references and are not subject to this restriction. Call
the selected registered tool in the same response instead of announcing a
future action. The episode is non-interactive and must not be restarted."""

BEGIN = """Follow the required read order, bind the current task's targets and
relations from fresh observation, then execute the first unmet recipe phase.
After each action verify its observable gate and preserve achieved relations.
For lingbot_act, provide only the chunk count; the runtime supplies the complete
current task_language."""


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Return the compact RoboTwin evaluation system prompt."""
    prompt = dict(base_evaluate.system_prompt(variables))
    prompt["READ ORDER"] = READ_ORDER
    prompt["VLA AND PRIMITIVE CONTROL"] = CONTROL
    prompt["RUNTIME"] = RUNTIME
    return prompt


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Return the compact RoboTwin evaluation user prompt."""
    return {"CELL": base_user.CELL, "BEGIN": BEGIN}
