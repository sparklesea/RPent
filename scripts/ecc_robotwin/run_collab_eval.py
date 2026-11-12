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

"""Evaluate Qwen+GPT collaborative RoboTwin control with stage-2 rules.

The Qwen worker reads the stage-1 task memory. The GPT-6 senior reads a
separate per-task memory rendered from the official RoboTwin GPT corpus and
receives it through the escalation correction hook. Both evaluation seeds are
run for all selected tasks, using the same stage-2 call rule for every
occurrence of a task.
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
RUNNER = Path(__file__).resolve().parent / "run_collab_eval_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.6-27b"
SENIOR_MODEL = "openai:gpt-6-astra"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
    write_json,
)


def load_runs(entry_indexes: list[int], task_limit: int | None) -> list[dict[str, Any]]:
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    task_names = list(evaluation["tasks"])
    if task_limit is not None:
        task_names = task_names[:task_limit]
    runs: list[dict[str, Any]] = []
    for entry_index in entry_indexes:
        for task_name in task_names:
            entries = evaluation["tasks"][task_name]
            if entry_index >= len(entries):
                raise ValueError(
                    f"task {task_name!r} has no entry {entry_index}; "
                    f"only {len(entries)} entries are available"
                )
            runs.append(
                {
                    "entry_index": entry_index,
                    "task_name": task_name,
                    "seed": int(entries[entry_index]["seed"]),
                    "task_language": entries[entry_index]["task_language"],
                }
            )
    expected_count = len(task_names) * len(entry_indexes)
    if len(runs) != expected_count:
        raise RuntimeError(f"expected {expected_count} runs, found {len(runs)}")
    return runs


def load_rules(rules_dir: Path) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for path in sorted(rules_dir.glob("*.json")):
        rule = json.loads(path.read_text(encoding="utf-8"))
        task = rule.get("task")
        if not task:
            raise ValueError(f"rule has no task: {path}")
        if task in rules:
            raise ValueError(f"duplicate rule for task {task!r}")
        required = {"gpt_plan", "checkin_K", "errors_N"}
        if not required.issubset(rule.get("rule") or {}):
            raise ValueError(f"rule is missing fields: {path}")
        rules[task] = rule
    return rules


def validate_inputs(
    runs: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
    qwen_memory_dir: Path,
    gpt_memory_source: Path,
) -> None:
    task_names = {run["task_name"] for run in runs}
    missing_rules = task_names - set(rules)
    if missing_rules:
        raise RuntimeError(f"stage-2 rules missing tasks: {sorted(missing_rules)}")

    qwen_task_memory = qwen_memory_dir / "task-specific"
    gpt_task_memory = gpt_memory_source / "task-specific"
    if not qwen_task_memory.is_dir():
        raise RuntimeError(f"Qwen memory directory not found: {qwen_task_memory}")
    if not gpt_task_memory.is_dir():
        raise RuntimeError(f"GPT memory directory not found: {gpt_task_memory}")
    for task_name in task_names:
        qwen_stem = f"robotwin_{task_name}_demo_randomized_s0"
        for suffix in (".json", "_recipe.jsonl"):
            if not (qwen_task_memory / f"{qwen_stem}{suffix}").is_file():
                raise RuntimeError(
                    f"Qwen memory reference missing: {qwen_stem}{suffix}"
                )
        gpt_audit = gpt_task_memory / f"{task_name}_s0.json"
        if not gpt_audit.is_file():
            raise RuntimeError(f"GPT memory audit missing: {gpt_audit}")


def render_gpt_memory(
    task_name: str,
    source: Path,
    destination: Path,
) -> None:
    audit = json.loads(source.read_text(encoding="utf-8"))
    sections = [
        f"# GPT-6 task memory: {task_name}",
        f"Task: {audit.get('task_language') or task_name}",
    ]
    semantic_recipe = audit.get("semantic_recipe")
    if semantic_recipe:
        sections.append(
            "## Semantic recipe\n"
            + json.dumps(semantic_recipe, ensure_ascii=False, indent=2)
        )
    strategy_notes = audit.get("strategy_notes")
    if strategy_notes:
        sections.append(
            "## Strategy notes\n"
            + json.dumps(strategy_notes, ensure_ascii=False, indent=2)
        )
    vla_recipe = audit.get("vla_recipe")
    if vla_recipe:
        sections.append(f"## Verified VLA guidance\n{vla_recipe}")
    what_failed = audit.get("what_failed")
    if what_failed:
        sections.append(
            "## Failure notes\n" + json.dumps(what_failed, ensure_ascii=False, indent=2)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def prepare_gpt_memories(
    run_dir: Path,
    gpt_memory_source: Path,
    task_names: set[str],
) -> dict[str, Path]:
    destination_root = run_dir / "gpt-memory"
    paths: dict[str, Path] = {}
    for task_name in sorted(task_names):
        source = gpt_memory_source / "task-specific" / f"{task_name}_s0.json"
        destination = destination_root / f"{task_name}.md"
        render_gpt_memory(task_name, source, destination)
        paths[task_name] = destination
    return paths


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
    run: dict[str, Any],
    rule: dict[str, Any],
    *,
    run_dir: Path,
    qwen_memory_dir: Path,
    gpt_memory_file: Path,
    gpu: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    task_name = run["task_name"]
    seed = run["seed"]
    entry_index = run["entry_index"]
    output_dir = run_dir / f"{task_name}_entry{entry_index}_s{seed}"
    result_path = output_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_NAME": task_name,
            "SEED": str(seed),
            "OUTPUT_DIR": str(output_dir),
            "QWEN_MEMORY_DIR": str(qwen_memory_dir),
            "GPT_MEMORY_FILE": str(gpt_memory_file),
            "GPU": gpu,
            "ROBOTWIN_ASSETS_PATH": settings["assets_path"],
            "LINGBOT_MODEL_PATH": settings["model_path"],
            "INFINI_BASE_URL": settings["worker_base_url"],
            "RPENT_BASE_URL": settings["senior_base_url"],
            "SESSION_TIMEOUT_S": str(settings["timeout_s"]),
            "TOOL_TIMEOUT_S": str(settings["timeout_s"]),
            "MAX_TURNS": str(settings["max_turns"]),
            "MAX_EPISODE_STEPS": str(settings["max_episode_steps"]),
            "RPENT_ESCALATION_PLAN": "1" if rule["rule"]["gpt_plan"] else "0",
            "RPENT_ESCALATION_CHECKIN": str(rule["rule"]["checkin_K"]),
            "RPENT_ESCALATION_ERRORS": str(rule["rule"]["errors_N"]),
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
        "phase": "qwen-gpt-collab-eval",
        "task": task_name,
        "entry_index": entry_index,
        "seed": seed,
        "task_language": run["task_language"],
        "gpu": gpu,
        "worker_model": WORKER_MODEL,
        "worker_reasoning_effort": "none",
        "senior_model": SENIOR_MODEL,
        "senior_reasoning_effort": "high",
        "rule": rule["rule"],
        "rule_config": rule["config"],
        "qwen_memory_dir": str(qwen_memory_dir.resolve()),
        "gpt_memory_file": str(gpt_memory_file.resolve()),
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


def update_status(
    run_dir: Path,
    rows: list[dict[str, Any]],
    total: int,
    entry_indexes: list[int],
) -> None:
    by_entry = {
        entry_index: {
            "finished": sum(row.get("entry_index") == entry_index for row in rows),
            "success": sum(
                row.get("entry_index") == entry_index and bool(row.get("solved"))
                for row in rows
            ),
            "gpt_calls": sum(
                row.get("entry_index") == entry_index and row.get("gpt_calls", 0)
                for row in rows
            ),
        }
        for entry_index in entry_indexes
    }
    write_json(
        run_dir / "status.json",
        {
            "phase": "qwen-gpt-collab-eval",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "task_config": "demo_randomized",
            "entry_indexes": entry_indexes,
            "planned_run_count": total,
            "finished_count": len(rows),
            "success_count": sum(bool(row.get("solved")) for row in rows),
            "gpt_call_count": sum(row.get("gpt_calls", 0) for row in rows),
            "subprocess_timeout_count": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            "by_entry_index": by_entry,
            "updated_at": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("rules_dir", type=Path)
    parser.add_argument("qwen_memory_dir", type=Path)
    parser.add_argument("gpt_memory_source", type=Path)
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--entry-indexes", default="2,3")
    parser.add_argument("--task-limit", type=int, default=None)
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
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("--task-limit must be positive")
    try:
        args.entry_indexes = [
            int(value) for value in args.entry_indexes.split(",") if value
        ]
    except ValueError as error:
        parser.error(f"--entry-indexes must be integers: {error}")
    if not args.entry_indexes:
        parser.error("--entry-indexes must not be empty")
    return args


def main() -> int:
    args = parse_args()
    if not RUNNER.is_file():
        raise RuntimeError(f"collaborative evaluation runner not found: {RUNNER}")
    if not Path(args.assets_path).is_dir():
        raise RuntimeError(f"RoboTwin assets not found: {args.assets_path}")
    if not Path(args.model_path).is_dir():
        raise RuntimeError(f"LingBot model not found: {args.model_path}")
    if not os.environ.get("INFINI_API_KEY"):
        raise RuntimeError("INFINI_API_KEY is required")
    if not os.environ.get("RPENT_API_KEY"):
        raise RuntimeError("RPENT_API_KEY is required")

    runs = load_runs(args.entry_indexes, args.task_limit)
    rules = load_rules(args.rules_dir)
    validate_inputs(runs, rules, args.qwen_memory_dir, args.gpt_memory_source)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    gpt_memory_files = prepare_gpt_memories(
        args.run_dir, args.gpt_memory_source, {run["task_name"] for run in runs}
    )

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
        "vla_endpoints": {},
    }
    write_json(
        args.run_dir / "batch.json",
        {
            "stage": "RoboTwin Qwen+GPT collaborative evaluation",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "entry_indexes": args.entry_indexes,
            "task_limit": args.task_limit,
            "rules_dir": str(args.rules_dir.resolve()),
            "qwen_memory_dir": str(args.qwen_memory_dir.resolve()),
            "gpt_memory_source": str(args.gpt_memory_source.resolve()),
            "timeout_s": args.timeout_s,
            "tool_timeout_s": args.timeout_s,
            "gpus": args.gpus.split(","),
            "workers_per_gpu": args.workers_per_gpu,
            "planned_run_count": len(runs),
            "started_at": utc_now(),
        },
    )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/result.json"))
    ]
    finished_runs = {(row["task"], row["entry_index"], row["seed"]) for row in rows}
    pending = [
        run
        for run in runs
        if (run["task_name"], run["entry_index"], run["seed"]) not in finished_runs
    ]
    update_status(args.run_dir, rows, len(runs), args.entry_indexes)

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
            for run in batch:
                future = executor.submit(
                    run_episode,
                    run,
                    rules[run["task_name"]],
                    run_dir=args.run_dir,
                    qwen_memory_dir=args.qwen_memory_dir,
                    gpt_memory_file=gpt_memory_files[run["task_name"]],
                    gpu=gpu,
                    settings=settings,
                )
                futures[future] = run

        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            update_status(args.run_dir, rows, len(runs), args.entry_indexes)
            print(
                json.dumps(
                    {
                        "finished": len(rows),
                        "total": len(runs),
                        "task": result["task"],
                        "entry_index": result["entry_index"],
                        "seed": result["seed"],
                        "success": bool(result["solved"]),
                        "gpt_calls": result.get("gpt_calls", 0),
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

    update_status(args.run_dir, rows, len(runs), args.entry_indexes)
    write_json(
        args.run_dir / "summary.json",
        {
            "runs": len(runs),
            "success": sum(bool(row.get("solved")) for row in rows),
            "gpt_calls": sum(row.get("gpt_calls", 0) for row in rows),
            "finished_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    main()
