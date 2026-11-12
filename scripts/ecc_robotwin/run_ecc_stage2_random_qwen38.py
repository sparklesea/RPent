#!/usr/bin/env python3
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

"""ECC RoboTwin Qwen 3.8 stage-2 ablation with random five-config search.

The primary stage-2 search tries a hand-ordered list of six cloud-call rules.
This ablation keeps the same six-attempt budget but randomizes the first five
configurations without replacement. ``GPT plan / K4 / N2`` is reserved as the
sixth and final fallback, so the fallback is always fresh and no configuration
is attempted twice for one task.

The candidate space is the full 12-cell Cartesian product of:

* plan: Qwen plan or GPT plan;
* K: 8 or 4;
* N: 4, 3, or 2.

For each task, a deterministic per-task RNG selects five configurations from
the eleven non-fallback cells. The fallback is appended last. The task stops at
the first successful configuration, or records the fallback result if every
attempt fails. All costs use the same GPT-6 price assumptions as the existing
stage-2 analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "robots" / "robotwin" / "eval" / "demo_randomized.json"
RUNNER = Path(__file__).resolve().parent / "run_ecc_stage2_qwen38_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.8-27b"
SENIOR_MODEL = "openai:gpt-6-astra"

CONFIGS = [
    (gpt_plan, checkin, errors)
    for checkin in (8, 4)
    for gpt_plan in (False, True)
    for errors in (4, 3, 2)
]
FALLBACK_CONFIG = (True, 4, 2)
RANDOM_CONFIG_COUNT = 5
MAX_CONFIGS = RANDOM_CONFIG_COUNT + 1

GROUP_RATIO = 2.38
INPUT_USD_PER_M = 10.0 * GROUP_RATIO
OUTPUT_USD_PER_M = 50.0 * GROUP_RATIO
CACHED_INPUT_USD_PER_M = 1.0 * GROUP_RATIO
CACHE_WRITE_USD_PER_M = 12.5 * GROUP_RATIO
LONG_CONTEXT_THRESHOLD = 272001
LONG_INPUT_USD_PER_M = 20.0 * GROUP_RATIO
LONG_OUTPUT_USD_PER_M = 75.0 * GROUP_RATIO
LONG_CACHED_INPUT_USD_PER_M = 2.0 * GROUP_RATIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
)


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact so concurrent readers see one version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    os.replace(temporary, path)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate an episode runner and every child it started, then reap it."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def load_tasks(entry_index: int, only: set[str] | None) -> list[dict[str, Any]]:
    """Load the selected RoboTwin task entries for rule search."""
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    tasks = []
    for task_name, entries in evaluation["tasks"].items():
        if only is not None and task_name not in only:
            continue
        if entry_index >= len(entries):
            raise ValueError(
                f"task {task_name!r} has no entry {entry_index}; "
                f"only {len(entries)} entries are available"
            )
        tasks.append(
            {
                "task_name": task_name,
                "seed": int(entries[entry_index]["seed"]),
                "task_language": entries[entry_index]["task_language"],
            }
        )
    if only is None and len(tasks) != 50:
        raise RuntimeError(f"expected 50 RoboTwin tasks, found {len(tasks)}")
    if only is not None:
        missing = only - {task["task_name"] for task in tasks}
        if missing:
            raise ValueError(f"unknown task names: {sorted(missing)}")
    return tasks


def validate_memory(memory_dir: Path, tasks: list[dict[str, Any]]) -> None:
    """Fail before GPU startup when any task-specific memory pair is absent."""
    task_specific = memory_dir / "task-specific"
    if not task_specific.is_dir():
        raise RuntimeError(f"memory task-specific directory not found: {task_specific}")
    for task in tasks:
        stem = f"robotwin_{task['task_name']}_demo_randomized_s0"
        for suffix in (".json", "_recipe.jsonl"):
            if not (task_specific / f"{stem}{suffix}").is_file():
                raise RuntimeError(f"stage-1 memory reference missing: {stem}{suffix}")


def config_name(config: tuple[bool, int, int]) -> str:
    """Return the canonical name used by rules and evaluators."""
    gpt_plan, checkin, errors = config
    return f"{'gptplan' if gpt_plan else 'qwenplan'}_K{checkin}_N{errors}"


def config_record(config: tuple[bool, int, int]) -> dict[str, Any]:
    """Return the rule fields for one configuration tuple."""
    gpt_plan, checkin, errors = config
    return {
        "gpt_plan": gpt_plan,
        "checkin_K": checkin,
        "errors_N": errors,
    }


def task_random_seed(random_seed: int, task: dict[str, Any]) -> str:
    """Derive a stable RNG seed that is independent for every task."""
    return f"ecc-stage2-random:{random_seed}:{task['task_name']}:{task['seed']}"


def random_search_order(
    random_seed: int, task: dict[str, Any]
) -> tuple[list[tuple[bool, int, int]], str]:
    """Sample five non-fallback configs and append the fixed fallback."""
    seed_text = task_random_seed(random_seed, task)
    rng = random.Random(seed_text)
    pool = [config for config in CONFIGS if config != FALLBACK_CONFIG]
    sampled = rng.sample(pool, RANDOM_CONFIG_COUNT)
    return [*sampled, FALLBACK_CONFIG], seed_text


def gpt_cost_usd(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
) -> float:
    """Price one request with the SoruxGPT GPT-6 group ratio."""
    cached_input_tokens = min(cached_input_tokens, input_tokens)
    if input_tokens > LONG_CONTEXT_THRESHOLD:
        input_rate = LONG_INPUT_USD_PER_M
        output_rate = LONG_OUTPUT_USD_PER_M
        cached_rate = LONG_CACHED_INPUT_USD_PER_M
    else:
        input_rate = INPUT_USD_PER_M
        output_rate = OUTPUT_USD_PER_M
        cached_rate = CACHED_INPUT_USD_PER_M
    uncached_input_tokens = input_tokens - cached_input_tokens
    return (
        uncached_input_tokens * input_rate
        + cached_input_tokens * cached_rate
        + cache_write_tokens * CACHE_WRITE_USD_PER_M
        + output_tokens * output_rate
    ) / 1_000_000


def parse_episode(output_dir: Path) -> dict[str, Any]:
    """Read task success, GPT usage, turns, and finish status."""
    states_path = output_dir / "states.json"
    environment_success = None
    if states_path.is_file():
        try:
            states = json.loads(states_path.read_text(encoding="utf-8"))
            environment_success = any(
                (step.get("result") or {}).get("episode_status", {}).get("eval_success")
                is True
                for step in reversed(states.get("steps") or [])
            )
        except (OSError, json.JSONDecodeError):
            environment_success = None

    gpt_calls = 0
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    cost = 0.0
    escalation_path = output_dir / "escalation.jsonl"
    if escalation_path.is_file():
        for line in escalation_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("role") != "senior":
                continue
            prompt = int(record.get("input_tokens") or 0)
            cached = int(record.get("cached_input_tokens") or 0)
            completion = int(record.get("output_tokens") or 0)
            gpt_calls += 1
            input_tokens += prompt
            cached_input_tokens += cached
            output_tokens += completion
            cost += gpt_cost_usd(prompt, cached, completion)

    turns = None
    finish_status = None
    transcripts = sorted(output_dir.glob("transcript_*.json"))
    if transcripts:
        try:
            transcript = json.loads(transcripts[-1].read_text(encoding="utf-8"))
            turns = (transcript.get("stats") or {}).get("turns_used")
            finish_status = (transcript.get("finish") or {}).get("status")
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "solved": environment_success is True,
        "environment_success": environment_success,
        "gpt_calls": gpt_calls,
        "gpt_input_tokens": input_tokens,
        "gpt_cached_input_tokens": cached_input_tokens,
        "gpt_output_tokens": output_tokens,
        "gpt_cost_usd": cost,
        "turns": turns,
        "finish_status": finish_status,
        "states": str(states_path) if states_path.exists() else None,
    }


def run_episode(
    task: dict[str, Any],
    config: tuple[bool, int, int],
    *,
    output_dir: Path,
    memory_dir: Path,
    gpu: str,
    settings: dict[str, Any],
    position: int,
    search_order: list[str],
    rng_seed: str,
) -> dict[str, Any]:
    """Run one configuration for one task, reusing a completed result."""
    result_path = output_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    gpt_plan, checkin, errors = config
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_NAME": task["task_name"],
            "SEED": str(task["seed"]),
            "OUTPUT_DIR": str(output_dir),
            "MEMORY_DIR": str(memory_dir),
            "GPU": gpu,
            "ROBOTWIN_ASSETS_PATH": settings["assets_path"],
            "LINGBOT_MODEL_PATH": settings["model_path"],
            "INFINI_BASE_URL": settings["worker_base_url"],
            "RPENT_BASE_URL": settings["senior_base_url"],
            "SESSION_TIMEOUT_S": str(settings["timeout_s"]),
            "TOOL_TIMEOUT_S": str(settings["timeout_s"]),
            "MAX_TURNS": str(settings["max_turns"]),
            "MAX_EPISODE_STEPS": str(settings["max_episode_steps"]),
            "RPENT_ESCALATION_PLAN": "1" if gpt_plan else "0",
            "RPENT_ESCALATION_CHECKIN": str(checkin),
            "RPENT_ESCALATION_ERRORS": str(errors),
            "VLA_ENDPOINT": settings["vla_endpoints"][gpu],
        }
    )
    if settings["rpent"]:
        environment["RPENT_BIN"] = settings["rpent"]

    started = time.monotonic()
    timed_out = False
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(
            [str(RUNNER)],
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=settings["subprocess_timeout_s"])
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
            stdout.write(
                f"\n[runner] subprocess timeout after "
                f"{settings['subprocess_timeout_s']}s\n"
            )
            _terminate_process_group(process)

    result = {
        "phase": "ecc-stage2-random-search",
        "task": task["task_name"],
        "seed": task["seed"],
        "task_language": task["task_language"],
        "gpu": gpu,
        "worker_model": WORKER_MODEL,
        "worker_reasoning_effort": "none",
        "senior_model": SENIOR_MODEL,
        "senior_reasoning_effort": "high",
        "config": config_record(config),
        "config_name": config_name(config),
        "sample_position": position,
        "is_fallback_position": config == FALLBACK_CONFIG,
        "random_seed": settings["random_seed"],
        "task_random_seed": rng_seed,
        "search_order": search_order,
        "planner_timeout_s": settings["timeout_s"],
        "tool_timeout_s": settings["timeout_s"],
        "exit_code": return_code,
        "subprocess_timeout": timed_out,
        "elapsed_s": round(time.monotonic() - started, 1),
        **parse_episode(output_dir),
        "finished_at": utc_now(),
    }
    write_json(result_path, result)
    return result


def rule_matches(existing: dict[str, Any], settings: dict[str, Any]) -> bool:
    """Return whether an existing rule belongs to this ablation run."""
    return (
        existing.get("algorithm") == "random5_plus_fallback"
        and existing.get("random_seed") == settings["random_seed"]
    )


def search_task(
    task: dict[str, Any],
    *,
    run_dir: Path,
    rules_dir: Path,
    memory_dir: Path,
    gpu: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Search one task with five random configs followed by the fallback."""
    task_name = task["task_name"]
    rule_path = rules_dir / f"{task_name}.json"
    if rule_path.exists():
        existing = json.loads(rule_path.read_text(encoding="utf-8"))
        if rule_matches(existing, settings):
            return existing
        rule_path.unlink()

    order, rng_seed = random_search_order(settings["random_seed"], task)
    order_names = [config_name(config) for config in order]
    task_dir = run_dir / f"{task_name}_s{task['seed']}"
    history: list[dict[str, Any]] = []
    for position, config in enumerate(order, start=1):
        output_dir = (
            task_dir
            / f"rs{settings['random_seed']}_s{position:02d}_{config_name(config)}"
        )
        result = run_episode(
            task,
            config,
            output_dir=output_dir,
            memory_dir=memory_dir,
            gpu=gpu,
            settings=settings,
            position=position,
            search_order=order_names,
            rng_seed=rng_seed,
        )
        history.append(
            {
                "config": config_name(config),
                **config_record(config),
                "sample_position": position,
                "sampled": position <= RANDOM_CONFIG_COUNT,
                "fallback": config == FALLBACK_CONFIG,
                "output_dir": str(output_dir),
                "success": bool(result.get("solved")),
                "gpt_calls": int(result.get("gpt_calls") or 0),
                "gpt_input_tokens": int(result.get("gpt_input_tokens") or 0),
                "gpt_cached_input_tokens": int(
                    result.get("gpt_cached_input_tokens") or 0
                ),
                "gpt_output_tokens": int(result.get("gpt_output_tokens") or 0),
                "gpt_cost_usd": float(result.get("gpt_cost_usd") or 0.0),
                "elapsed_s": float(result.get("elapsed_s") or 0.0),
                "subprocess_timeout": bool(result.get("subprocess_timeout")),
                "turns": result.get("turns"),
                "finish_status": result.get("finish_status"),
            }
        )
        if history[-1]["success"]:
            break

    success = next((item for item in history if item["success"]), None)
    chosen = success or history[-1]
    fallback_attempted = any(item["fallback"] for item in history)
    fallback_failed = fallback_attempted and not history[-1]["success"]
    rule = {
        "task": task_name,
        "search_seed": task["seed"],
        "rule": {
            "gpt_plan": chosen["gpt_plan"],
            "checkin_K": chosen["checkin_K"],
            "errors_N": chosen["errors_N"],
        },
        "config": chosen["config"],
        "algorithm": "random5_plus_fallback",
        "random_seed": settings["random_seed"],
        "task_random_seed": rng_seed,
        "sampled_configs": order_names[:-1],
        "search_order": order_names,
        "attempted_configs": [item["config"] for item in history],
        "fallback_config": config_name(FALLBACK_CONFIG),
        "fallback_attempted": fallback_attempted,
        "fallback_failed": fallback_failed,
        "solved_in_search": success is not None,
        "success_config": success["config"] if success else None,
        "success_position": success["sample_position"] if success else None,
        "episodes": len(history),
        "gpt_calls_in_search": sum(item["gpt_calls"] for item in history),
        "gpt_cost_usd_in_search": sum(item["gpt_cost_usd"] for item in history),
        "worker_elapsed_s_in_search": sum(item["elapsed_s"] for item in history),
        "success_threshold": 1,
        "history": history,
        "written_at": utc_now(),
    }
    write_json(rule_path, rule)
    return rule


def collect_rules(rules_dir: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Read completed rules that belong to this random-search run."""
    rules: list[dict[str, Any]] = []
    for path in sorted(rules_dir.glob("*.json")):
        try:
            rule = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # A worker may still be replacing the file; retry on the next loop.
            continue
        if rule_matches(rule, settings):
            rules.append(rule)
    return rules


def update_status(
    run_dir: Path,
    rules: list[dict[str, Any]],
    total: int,
    entry_index: int,
    settings: dict[str, Any],
) -> None:
    """Write the live aggregate status for dashboards and resumed runs."""
    history = [item for rule in rules for item in rule.get("history", [])]
    write_json(
        run_dir / "status.json",
        {
            "phase": "ecc-stage2-random-search",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "entry_index": entry_index,
            "task_config": "demo_randomized",
            "algorithm": "random5_plus_fallback",
            "random_seed": settings["random_seed"],
            "fallback_config": config_name(FALLBACK_CONFIG),
            "planned_task_count": total,
            "finished_count": len(rules),
            "solved_in_search_count": sum(
                bool(rule.get("solved_in_search")) for rule in rules
            ),
            "episode_count": sum(len(rule.get("history", [])) for rule in rules),
            "successful_episode_count": sum(
                bool(item.get("success")) for item in history
            ),
            "gpt_call_count": sum(
                int(rule.get("gpt_calls_in_search") or 0) for rule in rules
            ),
            "gpt_cost_usd": sum(
                float(rule.get("gpt_cost_usd_in_search") or 0.0) for rule in rules
            ),
            "worker_elapsed_s": sum(
                float(rule.get("worker_elapsed_s_in_search") or 0.0) for rule in rules
            ),
            "fallback_attempt_count": sum(
                bool(rule.get("fallback_attempted")) for rule in rules
            ),
            "fallback_failure_count": sum(
                bool(rule.get("fallback_failed")) for rule in rules
            ),
            "updated_at": utc_now(),
        },
    )


def update_summary(
    run_dir: Path,
    rules: list[dict[str, Any]],
    settings: dict[str, Any],
    started_at: str,
    wall_elapsed_s: float,
) -> None:
    """Write the final aggregate summary for the random-search ablation."""
    chosen_counts: dict[str, int] = {}
    for rule in rules:
        chosen_counts[rule["config"]] = chosen_counts.get(rule["config"], 0) + 1
    task_count = len(rules)
    episode_count = sum(len(rule.get("history", [])) for rule in rules)
    total_cost = sum(float(rule.get("gpt_cost_usd_in_search") or 0.0) for rule in rules)
    total_elapsed = sum(
        float(rule.get("worker_elapsed_s_in_search") or 0.0) for rule in rules
    )
    write_json(
        run_dir / "summary.json",
        {
            "tasks": task_count,
            "solved_in_search": sum(
                bool(rule.get("solved_in_search")) for rule in rules
            ),
            "episodes": episode_count,
            "average_configs_per_task": episode_count / task_count
            if task_count
            else 0.0,
            "gpt_calls_in_search": sum(
                int(rule.get("gpt_calls_in_search") or 0) for rule in rules
            ),
            "gpt_cost_usd": total_cost,
            "average_gpt_cost_usd_per_task": total_cost / task_count
            if task_count
            else 0.0,
            "worker_elapsed_s": total_elapsed,
            "average_worker_elapsed_s_per_task": total_elapsed / task_count
            if task_count
            else 0.0,
            "fallback_attempt_count": sum(
                bool(rule.get("fallback_attempted")) for rule in rules
            ),
            "fallback_failure_count": sum(
                bool(rule.get("fallback_failed")) for rule in rules
            ),
            "chosen_config_counts": dict(sorted(chosen_counts.items())),
            "algorithm": "random5_plus_fallback",
            "random_seed": settings["random_seed"],
            "sampled_config_count": RANDOM_CONFIG_COUNT,
            "fallback_config": config_name(FALLBACK_CONFIG),
            "config_space": [config_name(config) for config in CONFIGS],
            "gpt_pricing_usd_per_million": {
                "input": INPUT_USD_PER_M,
                "cached_input": CACHED_INPUT_USD_PER_M,
                "cache_write": CACHE_WRITE_USD_PER_M,
                "output": OUTPUT_USD_PER_M,
                "group_ratio": GROUP_RATIO,
            },
            "started_at": started_at,
            "wall_elapsed_s": wall_elapsed_s,
            "finished_at": utc_now(),
        },
    )


def runner_row(finished: int, total: int, rule: dict[str, Any]) -> dict[str, Any]:
    """Build one JSONL progress record for ``runner.log``."""
    return {
        "finished": finished,
        "total": total,
        "task": rule["task"],
        "random_seed": rule.get("random_seed"),
        "config": rule["config"],
        "solved_in_search": bool(rule.get("solved_in_search")),
        "episodes": int(rule.get("episodes") or 0),
        "gpt_calls": int(rule.get("gpt_calls_in_search") or 0),
        "gpt_cost_usd": float(rule.get("gpt_cost_usd_in_search") or 0.0),
        "worker_elapsed_s": float(rule.get("worker_elapsed_s_in_search") or 0.0),
        "fallback_attempted": bool(rule.get("fallback_attempted")),
        "fallback_failed": bool(rule.get("fallback_failed")),
        "search_order": rule.get("search_order"),
    }


def append_runner_log(path: Path, row: dict[str, Any]) -> None:
    """Append and flush one progress row for live monitoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def append_runner_error(path: Path, error: BaseException) -> None:
    """Persist a non-fatal progress aggregation failure for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "at": utc_now(),
                    "type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()


def logged_tasks(path: Path, random_seed: int) -> set[str]:
    """Return task names already present in a resumable runner log."""
    if not path.is_file():
        return set()
    tasks: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("random_seed") == random_seed:
            tasks.add(str(row.get("task")))
    return tasks


def elapsed_since(started_at: str) -> float:
    """Return wall-clock seconds since an ISO-8601 start timestamp."""
    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        return 0.0
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - start).total_seconds()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("rules_dir", type=Path)
    parser.add_argument("memory_dir", type=Path)
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--entry-index", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--subprocess-timeout-s", type=int, default=1200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--worker-base-url", default="https://cloud.infini-ai.com/maas")
    parser.add_argument("--senior-base-url", default="https://ai.soruxgpt.com")
    parser.add_argument("--rpent", default=None)
    parser.add_argument("--only", default=None, help="comma list of task names")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.timeout_s < 1 or args.subprocess_timeout_s < 1:
        parser.error("timeouts must be positive")
    if args.max_turns < 1 or args.max_episode_steps < 1:
        parser.error("--max-turns and --max-episode-steps must be positive")
    return args


def write_batch(
    run_dir: Path,
    args: argparse.Namespace,
    settings: dict[str, Any],
    started_at: str,
) -> None:
    """Write the immutable experiment configuration for this run."""
    write_json(
        run_dir / "batch.json",
        {
            "stage": "ECC RoboTwin stage 2 random-search ablation",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "entry_index": args.entry_index,
            "success_threshold": 1,
            "algorithm": "random5_plus_fallback",
            "random_seed": settings["random_seed"],
            "sampled_config_count": RANDOM_CONFIG_COUNT,
            "fallback_config": config_name(FALLBACK_CONFIG),
            "config_space": [config_name(config) for config in CONFIGS],
            "timeout_s": args.timeout_s,
            "tool_timeout_s": args.timeout_s,
            "gpus": settings["gpu_list"],
            "workers_per_gpu": args.workers_per_gpu,
            "memory_dir": str(args.memory_dir.resolve()),
            "rules_dir": str(args.rules_dir.resolve()),
            "started_at": started_at,
        },
    )


def main() -> int:
    args = parse_args()
    if not RUNNER.is_file():
        raise RuntimeError(f"stage-2 episode runner not found: {RUNNER}")
    if not Path(args.assets_path).is_dir():
        raise RuntimeError(f"RoboTwin assets not found: {args.assets_path}")
    if not Path(args.model_path).is_dir():
        raise RuntimeError(f"LingBot model not found: {args.model_path}")
    if not os.environ.get("INFINI_API_KEY"):
        raise RuntimeError("INFINI_API_KEY is required")
    if not os.environ.get("RPENT_API_KEY"):
        raise RuntimeError("RPENT_API_KEY is required")

    only = set(args.only.split(",")) if args.only else None
    tasks = load_tasks(args.entry_index, only)
    validate_memory(args.memory_dir, tasks)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.rules_dir.mkdir(parents=True, exist_ok=True)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    settings: dict[str, Any] = {
        "assets_path": str(Path(args.assets_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "worker_base_url": args.worker_base_url,
        "senior_base_url": args.senior_base_url,
        "timeout_s": args.timeout_s,
        "subprocess_timeout_s": args.subprocess_timeout_s,
        "max_turns": args.max_turns,
        "max_episode_steps": args.max_episode_steps,
        "rpent": args.rpent,
        "gpu_list": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "random_seed": args.random_seed,
        "vla_endpoints": {},
    }
    batch_path = args.run_dir / "batch.json"
    started_at = utc_now()
    if batch_path.is_file():
        try:
            started_at = (
                json.loads(batch_path.read_text()).get("started_at") or started_at
            )
        except (OSError, json.JSONDecodeError):
            pass
    write_batch(args.run_dir, args, settings, started_at)
    rules = collect_rules(args.rules_dir, settings)
    finished_tasks = {rule["task"] for rule in rules if rule_matches(rule, settings)}
    pending = [task for task in tasks if task["task_name"] not in finished_tasks]
    update_status(args.run_dir, rules, len(tasks), args.entry_index, settings)

    runner_log = args.run_dir / "runner.log"
    already_logged = logged_tasks(runner_log, settings["random_seed"])
    shared_vlas: dict[str, tuple[subprocess.Popen, str]] = {}
    executors = []
    futures = {}
    try:
        if pending:
            for gpu in gpus:
                shared_vlas[gpu] = start_shared_vla(
                    gpu, Path(settings["model_path"]), args.run_dir
                )
                settings["vla_endpoints"][gpu] = shared_vlas[gpu][1]

        batches = {gpu: pending[index :: len(gpus)] for index, gpu in enumerate(gpus)}
        for gpu, batch in batches.items():
            executor = ThreadPoolExecutor(
                max_workers=settings["workers_per_gpu"],
                thread_name_prefix=f"gpu{gpu}",
            )
            executors.append(executor)
            for task in batch:
                future = executor.submit(
                    search_task,
                    task,
                    run_dir=args.run_dir,
                    rules_dir=args.rules_dir,
                    memory_dir=args.memory_dir,
                    gpu=gpu,
                    settings=settings,
                )
                futures[future] = task

        pending_futures = set(futures)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures, timeout=30, return_when=FIRST_COMPLETED
            )
            for future in done:
                try:
                    rule = future.result()
                except Exception as error:
                    task = futures[future]
                    print(
                        json.dumps(
                            {
                                "task": task["task_name"],
                                "seed": task["seed"],
                                "runner_error": f"{type(error).__name__}: {error}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                try:
                    rules = collect_rules(args.rules_dir, settings)
                    update_status(
                        args.run_dir, rules, len(tasks), args.entry_index, settings
                    )
                    if rule["task"] not in already_logged:
                        append_runner_log(
                            runner_log, runner_row(len(rules), len(tasks), rule)
                        )
                        already_logged.add(rule["task"])
                    print(
                        json.dumps(
                            {
                                "finished": len(rules),
                                "total": len(tasks),
                                "task": rule["task"],
                                "config": rule["config"],
                                "solved_in_search": rule["solved_in_search"],
                                "episodes": rule["episodes"],
                                "gpt_calls": rule["gpt_calls_in_search"],
                                "gpt_cost_usd": round(
                                    rule["gpt_cost_usd_in_search"], 4
                                ),
                                "fallback_attempted": rule["fallback_attempted"],
                                "fallback_failed": rule["fallback_failed"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception as error:
                    append_runner_error(args.run_dir / "runner_errors.log", error)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        for process, _ in shared_vlas.values():
            stop_shared_vla(process)

    rules = collect_rules(args.rules_dir, settings)
    update_status(args.run_dir, rules, len(tasks), args.entry_index, settings)
    update_summary(
        args.run_dir,
        rules,
        settings,
        started_at,
        round(elapsed_since(started_at), 1),
    )
    return 0


if __name__ == "__main__":
    main()
