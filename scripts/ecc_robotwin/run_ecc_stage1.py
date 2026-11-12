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

"""ECC RoboTwin stage 1: device-side task-memory exploration.

Qwen3.6-27B alone explores all 50 ``demo_randomized`` tasks using the first
official seed entry for each task. Each task uses independent exploration
sessions with a per-session planner timeout. The source memory is copied to
``memory_dir`` so the official corpus is never mutated.

The official corpus stores task recipes as ``<task>_s0`` while local RoboTwin
prompts reference ``robotwin_<task>_demo_randomized_s0``. Initialization
creates the latter as compatibility aliases. A solved exploration replaces
both slots with its audit/recipe pair and archives the official files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "robots" / "robotwin" / "eval" / "demo_randomized.json"
DEFAULT_SOURCE_MEMORY = ROOT / "memory" / "robotwin"
RUNNER = Path(__file__).resolve().parent / "run_explore_qwen.sh"
SHARED_VLA_RUNNER = Path(__file__).resolve().parent / "run_shared_vla.sh"
MODEL_ID = "openai-chat:qwen3.6-27b"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def pick_free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def start_shared_vla(
    gpu: str, model_path: Path, run_dir: Path
) -> tuple[subprocess.Popen, str]:
    port = pick_free_port()
    log_path = run_dir / f"shared_vla_gpu{gpu}.log"
    command = [str(SHARED_VLA_RUNNER), gpu, str(port), str(model_path)]
    environment = os.environ.copy()
    environment["PYTHON_BIN"] = sys.executable
    log_stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.monotonic() + 900.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_stream.close()
            raise RuntimeError(
                f"shared VLA on GPU {gpu} exited with code {process.returncode}; "
                f"see {log_path}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                log_stream.close()
                return process, f"ws://127.0.0.1:{port}"
        except OSError:
            time.sleep(1.0)
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
    log_stream.close()
    raise RuntimeError(f"shared VLA on GPU {gpu} did not listen within 900s")


def stop_shared_vla(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
    if process.stdin is not None:
        process.stdin.close()


def load_tasks(entry_index: int, only: set[str] | None) -> list[dict[str, Any]]:
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
    if len(tasks) != 50 and only is None:
        raise RuntimeError(f"expected 50 RoboTwin tasks, found {len(tasks)}")
    return tasks


def initialize_memory(
    memory_dir: Path,
    source_memory: Path,
    tasks: list[dict[str, Any]],
) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    if not any(memory_dir.iterdir()):
        shutil.copytree(source_memory, memory_dir, dirs_exist_ok=True)

    task_specific = memory_dir / "task-specific"
    task_specific.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_name = task["task_name"]
        official_stem = f"{task_name}_s0"
        prompt_stem = f"robotwin_{task_name}_demo_randomized_s0"
        for suffix in (".json", "_recipe.jsonl"):
            official = task_specific / f"{official_stem}{suffix}"
            alias = task_specific / f"{prompt_stem}{suffix}"
            if official.is_file() and not alias.exists():
                shutil.copy2(official, alias)


def recipe_paths(output_dir: Path, task_name: str, seed: int) -> tuple[Path, Path]:
    tag = f"robotwin_{task_name}_demo_randomized_s{seed}"
    return output_dir / f"{tag}.json", output_dir / f"{tag}_recipe.jsonl"


def parse_episode(output_dir: Path, task_name: str, seed: int) -> dict[str, Any]:
    audit, recipe = recipe_paths(output_dir, task_name, seed)
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
    return {
        "solved": audit.is_file() and recipe.is_file(),
        "environment_success": environment_success,
        "audit": str(audit) if audit.exists() else None,
        "recipe": str(recipe) if recipe.exists() else None,
    }


def publish_memory(
    memory_dir: Path,
    output_dir: Path,
    task_name: str,
    seed: int,
) -> dict[str, Any]:
    audit, recipe = recipe_paths(output_dir, task_name, seed)
    if not audit.is_file() or not recipe.is_file():
        return {"replaced": False, "reason": "successful audit/recipe pair missing"}

    task_specific = memory_dir / "task-specific"
    archive = memory_dir / "_internal" / "replaced" / "official" / task_name
    archive.mkdir(parents=True, exist_ok=True)
    targets = {
        f"robotwin_{task_name}_demo_randomized_s0.json": audit,
        f"robotwin_{task_name}_demo_randomized_s0_recipe.jsonl": recipe,
        f"{task_name}_s0.json": audit,
        f"{task_name}_s0_recipe.jsonl": recipe,
    }
    for target_name, source in targets.items():
        target = task_specific / target_name
        if target.exists():
            shutil.copy2(target, archive / f"official-{target_name}")
        shutil.copy2(source, target)
    return {
        "replaced": True,
        "targets": sorted(targets),
        "archive": str(archive),
    }


def run_task(
    task: dict[str, Any],
    *,
    run_dir: Path,
    memory_dir: Path,
    gpu: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    task_name = task["task_name"]
    seed = task["seed"]
    output_dir = run_dir / f"{task_name}_s{seed}"
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
            "MEMORY_DIR": str(memory_dir),
            "GPU": gpu,
            "ROBOTWIN_ASSETS_PATH": settings["assets_path"],
            "LINGBOT_MODEL_PATH": settings["model_path"],
            "INFINI_BASE_URL": settings["base_url"],
            "EXPLORE_SESSIONS": str(settings["sessions"]),
            "EXPLORE_ATTEMPTS": str(settings["attempts"]),
            "SESSION_TIMEOUT_S": str(settings["planner_timeout_s"]),
            "TOOL_TIMEOUT_S": str(settings["tool_timeout_s"]),
            "MAX_TURNS": str(settings["max_turns"]),
            "MAX_EPISODE_STEPS": str(settings["max_episode_steps"]),
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

    parsed = parse_episode(output_dir, task_name, seed)
    publication = (
        publish_memory(memory_dir, output_dir, task_name, seed)
        if parsed["solved"]
        else {
            "replaced": False,
            "reason": "task exploration did not produce successful artifacts",
        }
    )
    result = {
        "phase": 1,
        "task": task_name,
        "seed": seed,
        "task_language": task["task_language"],
        "gpu": gpu,
        "model": MODEL_ID,
        "reasoning_effort": "none",
        "planner_timeout_s": settings["planner_timeout_s"],
        "tool_timeout_s": settings["tool_timeout_s"],
        "explore_sessions": settings["sessions"],
        "explore_attempts_per_session": settings["attempts"],
        "exit_code": return_code,
        "subprocess_timeout": timed_out,
        "elapsed_s": round(time.monotonic() - started, 1),
        **parsed,
        "memory_publication": publication,
        "finished_at": utc_now(),
    }
    write_json(result_path, result)
    return result


def update_status(
    run_dir: Path,
    rows: list[dict[str, Any]],
    total: int,
    settings: dict[str, Any],
) -> None:
    write_json(
        run_dir / "status.json",
        {
            "phase": 1,
            "model": MODEL_ID,
            "reasoning_effort": "none",
            "planner_timeout_s": settings["planner_timeout_s"],
            "tool_timeout_s": settings["tool_timeout_s"],
            "task_config": "demo_randomized",
            "planned_task_count": total,
            "finished_count": len(rows),
            "success_count": sum(bool(row.get("solved")) for row in rows),
            "updated_at": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("memory_dir", type=Path)
    parser.add_argument("--source-memory", type=Path, default=DEFAULT_SOURCE_MEMORY)
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--entry-index", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--planner-timeout-s", type=int, default=1200)
    parser.add_argument("--tool-timeout-s", type=int, default=1200)
    parser.add_argument("--subprocess-timeout-s", type=int, default=6600)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--base-url", default="https://cloud.infini-ai.com/maas")
    parser.add_argument("--rpent", default=None)
    parser.add_argument("--only", default=None, help="comma list of task names")
    args = parser.parse_args()
    if args.sessions < 1 or args.attempts < 0:
        parser.error("--sessions must be positive and --attempts nonnegative")
    if args.planner_timeout_s < 1 or args.tool_timeout_s < 1:
        parser.error("timeouts must be positive")
    if args.subprocess_timeout_s < args.planner_timeout_s * args.sessions:
        parser.error("--subprocess-timeout-s must cover all exploration sessions")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not RUNNER.is_file():
        raise RuntimeError(f"exploration runner not found: {RUNNER}")
    if not args.source_memory.is_dir():
        raise RuntimeError(f"source memory not found: {args.source_memory}")
    if not Path(args.assets_path).is_dir():
        raise RuntimeError(f"RoboTwin assets not found: {args.assets_path}")
    if not Path(args.model_path).is_dir():
        raise RuntimeError(f"LingBot model not found: {args.model_path}")
    if not os.environ.get("INFINI_API_KEY"):
        raise RuntimeError("INFINI_API_KEY is required")

    only = set(args.only.split(",")) if args.only else None
    tasks = load_tasks(args.entry_index, only)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    initialize_memory(args.memory_dir, args.source_memory, tasks)

    settings = {
        "assets_path": str(Path(args.assets_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "base_url": args.base_url,
        "sessions": args.sessions,
        "attempts": args.attempts,
        "planner_timeout_s": args.planner_timeout_s,
        "tool_timeout_s": args.tool_timeout_s,
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
            "stage": "ECC RoboTwin stage 1 device-side exploration",
            "model": MODEL_ID,
            "reasoning_effort": "none",
            "entry_index": args.entry_index,
            "sessions": args.sessions,
            "attempts": args.attempts,
            "planner_timeout_s": args.planner_timeout_s,
            "tool_timeout_s": args.tool_timeout_s,
            "gpus": args.gpus.split(","),
            "workers_per_gpu": args.workers_per_gpu,
            "memory_dir": str(args.memory_dir.resolve()),
            "source_memory": str(args.source_memory.resolve()),
            "started_at": utc_now(),
            "only": sorted(only) if only else None,
        },
    )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/result.json"))
    ]
    finished_tasks = {row["task"] for row in rows}
    pending = [task for task in tasks if task["task_name"] not in finished_tasks]
    update_status(args.run_dir, rows, len(tasks), settings)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    batches = {gpu: pending[index :: len(gpus)] for index, gpu in enumerate(gpus)}
    task_by_name = {task["task_name"]: task for task in tasks}
    gpu_by_task = {
        task["task_name"]: gpu for gpu, batch in batches.items() for task in batch
    }
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
                max_workers=settings["workers_per_gpu"],
                thread_name_prefix=f"gpu{gpu}",
            )
            executors.append(executor)
            for task in batch:
                future = executor.submit(
                    run_task,
                    task,
                    run_dir=args.run_dir,
                    memory_dir=args.memory_dir,
                    gpu=gpu,
                    settings=settings,
                )
                futures[future] = task["task_name"]
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                task = task_by_name[task_name]
                row = {
                    "phase": 1,
                    "task": task_name,
                    "seed": task["seed"],
                    "task_language": task["task_language"],
                    "gpu": gpu_by_task[task_name],
                    "model": MODEL_ID,
                    "reasoning_effort": "none",
                    "planner_timeout_s": settings["planner_timeout_s"],
                    "tool_timeout_s": settings["tool_timeout_s"],
                    "exit_code": None,
                    "subprocess_timeout": False,
                    "elapsed_s": None,
                    "solved": False,
                    "runner_error": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                }
            rows.append(row)
            update_status(args.run_dir, rows, len(tasks), settings)
            print(
                json.dumps(
                    {
                        "finished": len(rows),
                        "total": len(tasks),
                        "task": task_name,
                        "success": bool(row.get("solved")),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        for executor in executors:
            executor.shutdown(cancel_futures=True)
        for process, _ in shared_vlas.values():
            stop_shared_vla(process)

    write_json(
        args.run_dir / "summary.json",
        {
            "tasks": len(rows),
            "success": sum(bool(row.get("solved")) for row in rows),
            "finished_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
