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

"""ECC RoboTwin stage 2: search per-task cloud-call rules.

Each task uses the stage-1 memory read-only and tries configurations from the
least to most GPT-heavy escalation rule. A configuration is accepted only after
it succeeds `success_threshold` times on the same seed; if none reaches the
threshold, the strongest configuration is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "robots" / "robotwin" / "eval" / "demo_randomized.json"
RUNNER = Path(__file__).resolve().parent / "run_ecc_stage2_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.6-27b"
SENIOR_MODEL = "openai:gpt-6-astra"
CONFIGS = [
    (False, 8, 4),
    (True, 8, 4),
    (False, 4, 4),
    (True, 4, 4),
    (True, 4, 3),
    (True, 4, 2),
]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
    write_json,
)


def config_name(index: int, config: tuple[bool, int, int]) -> str:
    gpt_plan, checkin, errors = config
    return f"c{index + 1}_{'gptplan' if gpt_plan else 'qwenplan'}_K{checkin}_N{errors}"


def load_tasks(entry_index: int) -> list[dict[str, Any]]:
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    tasks = []
    for task_name, entries in evaluation["tasks"].items():
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
    if len(tasks) != 50:
        raise RuntimeError(f"expected 50 RoboTwin tasks, found {len(tasks)}")
    return tasks


def validate_memory(memory_dir: Path, tasks: list[dict[str, Any]]) -> None:
    task_specific = memory_dir / "task-specific"
    if not task_specific.is_dir():
        raise RuntimeError(f"memory task-specific directory not found: {task_specific}")
    for task in tasks:
        stem = f"robotwin_{task['task_name']}_demo_randomized_s0"
        for suffix in (".json", "_recipe.jsonl"):
            if not (task_specific / f"{stem}{suffix}").is_file():
                raise RuntimeError(f"stage-1 memory reference missing: {stem}{suffix}")


def parse_episode(output_dir: Path) -> dict[str, Any]:
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
    escalation_path = output_dir / "escalation.jsonl"
    if escalation_path.is_file():
        try:
            gpt_calls = sum(
                json.loads(line).get("role") == "senior"
                for line in escalation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, json.JSONDecodeError):
            gpt_calls = 0

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
        "turns": turns,
        "finish_status": finish_status,
        "states": str(states_path) if states_path.exists() else None,
    }


def run_episode(
    task: dict[str, Any],
    config: tuple[bool, int, int],
    *,
    run_dir: Path,
    memory_dir: Path,
    gpu: str,
    settings: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    gpt_plan, checkin, errors = config
    output_dir = (
        run_dir
        / f"{task['task_name']}_s{task['seed']}"
        / config_name(CONFIGS.index(config), config)
    )
    if settings["success_threshold"] > 1:
        output_dir = (
            output_dir
            / f"threshold_{settings['success_threshold']}"
            / f"attempt_{attempt}"
        )
    result_path = output_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
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
        try:
            completed = subprocess.run(
                [str(RUNNER)],
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                timeout=settings["subprocess_timeout_s"],
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
            stdout.write(
                f"\n[runner] subprocess timeout after "
                f"{settings['subprocess_timeout_s']}s\n"
            )

    result = {
        "phase": 2,
        "task": task["task_name"],
        "seed": task["seed"],
        "task_language": task["task_language"],
        "gpu": gpu,
        "worker_model": WORKER_MODEL,
        "worker_reasoning_effort": "none",
        "senior_model": SENIOR_MODEL,
        "senior_reasoning_effort": "high",
        "config": {
            "gpt_plan": gpt_plan,
            "checkin_K": checkin,
            "errors_N": errors,
        },
        "success_threshold": settings["success_threshold"],
        "attempt": attempt,
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


def search_task(
    task: dict[str, Any],
    *,
    run_dir: Path,
    rules_dir: Path,
    memory_dir: Path,
    gpu: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    task_name = task["task_name"]
    rule_path = rules_dir / f"{task_name}.json"
    if rule_path.exists():
        existing_rule = json.loads(rule_path.read_text(encoding="utf-8"))
        if existing_rule.get("success_threshold", 1) == settings["success_threshold"]:
            return existing_rule
        rule_path.unlink()

    history: list[dict[str, Any]] = []
    for index, config in enumerate(CONFIGS):
        name = config_name(index, config)
        attempt_results = []
        successes = 0
        for attempt in range(1, settings["success_threshold"] + 1):
            result = run_episode(
                task,
                config,
                run_dir=run_dir,
                memory_dir=memory_dir,
                gpu=gpu,
                settings=settings,
                attempt=attempt,
            )
            attempt_results.append(result)
            if not result.get("solved"):
                break
            successes += 1

        config_success = successes >= settings["success_threshold"]
        history.append(
            {
                "config": name,
                "success": config_success,
                "attempts": len(attempt_results),
                "successes": successes,
                "gpt_calls": sum(
                    result.get("gpt_calls", 0) for result in attempt_results
                ),
                "turns": [result.get("turns") for result in attempt_results],
                "finish_status": [
                    result.get("finish_status") for result in attempt_results
                ],
                "subprocess_timeout": any(
                    bool(result.get("subprocess_timeout")) for result in attempt_results
                ),
            }
        )
        if config_success:
            break

    chosen = next((item for item in history if item["success"]), None)
    chosen_index = (
        next(
            index
            for index, item in enumerate(history)
            if item["config"] == chosen["config"]
        )
        if chosen is not None
        else len(CONFIGS) - 1
    )
    rule = {
        "task": task_name,
        "search_seed": task["seed"],
        "rule": {
            "gpt_plan": CONFIGS[chosen_index][0],
            "checkin_K": CONFIGS[chosen_index][1],
            "errors_N": CONFIGS[chosen_index][2],
        },
        "config": config_name(chosen_index, CONFIGS[chosen_index]),
        "solved_in_search": chosen is not None,
        "episodes": sum(item.get("attempts", 1) for item in history),
        "gpt_calls_in_search": sum(item["gpt_calls"] for item in history),
        "success_threshold": settings["success_threshold"],
        "history": history,
        "written_at": utc_now(),
    }
    write_json(rule_path, rule)
    return rule


def collect_rules(rules_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rules_dir.glob("*.json"))
    ]


def update_status(
    run_dir: Path,
    rules: list[dict[str, Any]],
    total: int,
    entry_index: int,
) -> None:
    episodes = [item for rule in rules for item in rule.get("history", [])]
    write_json(
        run_dir / "status.json",
        {
            "phase": 2,
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "entry_index": entry_index,
            "task_config": "demo_randomized",
            "planned_task_count": total,
            "finished_count": len(rules),
            "solved_in_search_count": sum(
                bool(rule.get("solved_in_search")) for rule in rules
            ),
            "episode_count": sum(item.get("attempts", 1) for item in episodes),
            "successful_episode_count": sum(
                item.get("successes", 1 if item.get("success") else 0)
                for item in episodes
            ),
            "gpt_call_count": sum(rule.get("gpt_calls_in_search", 0) for rule in rules),
            "updated_at": utc_now(),
        },
    )


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
    parser.add_argument("--success-threshold", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--subprocess-timeout-s", type=int, default=1200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--worker-base-url", default="https://cloud.infini-ai.com/maas")
    parser.add_argument("--senior-base-url", default="https://ai.soruxgpt.com")
    parser.add_argument("--rpent", default=None)
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.timeout_s < 1 or args.subprocess_timeout_s < 1:
        parser.error("timeouts must be positive")
    if args.max_turns < 1 or args.max_episode_steps < 1:
        parser.error("--max-turns and --max-episode-steps must be positive")
    if args.success_threshold < 1:
        parser.error("--success-threshold must be positive")
    return args


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

    tasks = load_tasks(args.entry_index)
    validate_memory(args.memory_dir, tasks)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.rules_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "assets_path": str(Path(args.assets_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "worker_base_url": args.worker_base_url,
        "senior_base_url": args.senior_base_url,
        "timeout_s": args.timeout_s,
        "subprocess_timeout_s": args.subprocess_timeout_s,
        "max_turns": args.max_turns,
        "max_episode_steps": args.max_episode_steps,
        "rpent": args.rpent,
        "workers_per_gpu": args.workers_per_gpu,
        "success_threshold": args.success_threshold,
        "vla_endpoints": {},
    }
    write_json(
        args.run_dir / "batch.json",
        {
            "stage": "ECC RoboTwin stage 2 call-rule search",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "entry_index": args.entry_index,
            "success_threshold": args.success_threshold,
            "configs": [
                config_name(index, config) for index, config in enumerate(CONFIGS)
            ],
            "timeout_s": args.timeout_s,
            "tool_timeout_s": args.timeout_s,
            "gpus": args.gpus.split(","),
            "workers_per_gpu": args.workers_per_gpu,
            "memory_dir": str(args.memory_dir.resolve()),
            "rules_dir": str(args.rules_dir.resolve()),
            "started_at": utc_now(),
        },
    )

    rules = collect_rules(args.rules_dir)
    finished_tasks = {
        rule["task"]
        for rule in rules
        if rule.get("success_threshold", 1) == args.success_threshold
    }
    pending = [task for task in tasks if task["task_name"] not in finished_tasks]
    update_status(args.run_dir, rules, len(tasks), args.entry_index)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    batches = {gpu: pending[index :: len(gpus)] for index, gpu in enumerate(gpus)}
    shared_vlas: dict[str, tuple[subprocess.Popen, str]] = {}
    executors = []
    futures = {}
    try:
        for gpu in gpus:
            shared_vlas[gpu] = start_shared_vla(
                gpu, Path(settings["model_path"]), args.run_dir
            )
            settings["vla_endpoints"][gpu] = shared_vlas[gpu][1]
        for gpu, batch in batches.items():
            executor = ThreadPoolExecutor(
                max_workers=settings["workers_per_gpu"], thread_name_prefix=f"gpu{gpu}"
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

        for future in as_completed(futures):
            rule = future.result()
            rules = collect_rules(args.rules_dir)
            update_status(args.run_dir, rules, len(tasks), args.entry_index)
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
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        for process, _ in shared_vlas.values():
            stop_shared_vla(process)

    rules = collect_rules(args.rules_dir)
    update_status(args.run_dir, rules, len(tasks), args.entry_index)
    chosen_counts: dict[str, int] = {}
    for rule in rules:
        chosen_counts[rule["config"]] = chosen_counts.get(rule["config"], 0) + 1
    write_json(
        args.run_dir / "summary.json",
        {
            "tasks": len(rules),
            "solved_in_search": sum(
                bool(rule.get("solved_in_search")) for rule in rules
            ),
            "episodes": sum(rule.get("episodes", 0) for rule in rules),
            "gpt_calls_in_search": sum(
                rule.get("gpt_calls_in_search", 0) for rule in rules
            ),
            "chosen_config_counts": dict(sorted(chosen_counts.items())),
            "finished_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    main()
