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

"""Evaluate per-turn Qwen routing between Qwen and GPT on RoboTwin.

Every planner request first asks Qwen for a route. Qwen then handles the action
when the decision is ``qwen``; GPT handles it when the decision is ``gpt``. The
Qwen branch uses the stage-2 local memory profile, while the GPT branch receives
the same per-task memory used by the previous collaborative evaluations.
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
RUNNER = Path(__file__).resolve().parent / "run_qwen_router_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.6-27b"
GPT_MODEL = "openai:gpt-6-astra"
DEFAULT_QWEN_MEMORY = ROOT / "runs" / "robotwin-ecc-stage2-gpt-memory" / "memory"
DEFAULT_GPT_MEMORY_SOURCE = ROOT / "memory" / "robotwin"
DEFAULT_ASSETS = ROOT / "runs" / "robotwin-ecc-stage1" / "assets"
DEFAULT_MODEL = (
    ROOT
    / "runs"
    / "robotwin-ecc-stage1"
    / "models"
    / "LingBot-VLA-RoboTwin-EEF-ckpt1500"
)
PHASE = "qwen-router-gpt-memory-eval"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_candidate_verify_eval import validate_memory  # noqa: E402
from run_collab_eval import load_runs, prepare_gpt_memories  # noqa: E402
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
    write_json,
)


def parse_router_log(output_dir: Path) -> dict[str, Any]:
    """Aggregate router decisions and model calls from one episode."""
    path = output_dir / "qwen_router.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    router_calls = len(rows)
    selected_qwen = sum(row.get("selected") == "qwen" for row in rows)
    selected_gpt = sum(row.get("selected") == "gpt" for row in rows)
    return {
        "router_requests": router_calls,
        "qwen_calls": router_calls + selected_qwen,
        "gpt_calls": selected_gpt,
        "qwen_route_count": selected_qwen,
        "gpt_route_count": selected_gpt,
        "route_error_count": sum(bool(row.get("route_error")) for row in rows),
        "route_fallback_count": sum(bool(row.get("route_fallback")) for row in rows),
        "qwen_action_error_count": sum(bool(row.get("qwen_error")) for row in rows),
        "gpt_action_error_count": sum(bool(row.get("gpt_error")) for row in rows),
    }


def parse_episode(output_dir: Path) -> dict[str, Any]:
    """Read task success, turn count, and router statistics."""
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
        "turns": turns,
        "finish_status": finish_status,
        "states": str(states_path) if states_path.exists() else None,
        **parse_router_log(output_dir),
    }


def run_episode(
    run: dict[str, Any],
    *,
    run_dir: Path,
    qwen_memory_dir: Path,
    gpt_memory_file: Path,
    gpu: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Run or resume one task/seed episode."""
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
            "RPENT_BASE_URL": settings["gpt_base_url"],
            "SESSION_TIMEOUT_S": str(settings["timeout_s"]),
            "TOOL_TIMEOUT_S": str(settings["timeout_s"]),
            "MAX_TURNS": str(settings["max_turns"]),
            "MAX_EPISODE_STEPS": str(settings["max_episode_steps"]),
            "RPENT_QWEN_ROUTER_GPT_MODEL": settings["gpt_model"],
            "RPENT_QWEN_ROUTER_GPT_REASONING_EFFORT": settings["gpt_reasoning_effort"],
            "RPENT_QWEN_ROUTER_TEMPERATURE": str(settings["router_temperature"]),
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
        "phase": PHASE,
        "task": task_name,
        "entry_index": entry_index,
        "seed": seed,
        "task_language": run["task_language"],
        "gpu": gpu,
        "worker_model": WORKER_MODEL,
        "worker_reasoning_effort": "none",
        "router_model": WORKER_MODEL,
        "router_temperature": settings["router_temperature"],
        "gpt_model": settings["gpt_model"],
        "gpt_reasoning_effort": settings["gpt_reasoning_effort"],
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


def _averages(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize calls over successful runs."""
    successful = [row for row in rows if row.get("solved")]
    count = len(successful)

    def average(key: str, *, digits: int = 2) -> float:
        if not count:
            return 0.0
        return round(sum(row.get(key, 0) for row in successful) / count, digits)

    return {
        "success_count": count,
        "avg_router_calls_per_success": average("router_requests"),
        "avg_qwen_calls_per_success": average("qwen_calls"),
        "avg_gpt_calls_per_success": average("gpt_calls"),
        "avg_gpt_route_count_per_success": average("gpt_route_count"),
        "avg_turns_per_success": average("turns"),
        "avg_elapsed_s_per_success": average("elapsed_s", digits=1),
    }


def update_status(
    run_dir: Path,
    rows: list[dict[str, Any]],
    total: int,
    settings: dict[str, Any],
) -> None:
    """Write resumable batch status and accuracy/call summaries."""
    write_json(
        run_dir / "status.json",
        {
            "phase": PHASE,
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "router_model": WORKER_MODEL,
            "router_temperature": settings["router_temperature"],
            "gpt_model": settings["gpt_model"],
            "gpt_reasoning_effort": settings["gpt_reasoning_effort"],
            "task_config": "demo_randomized",
            "entry_indexes": settings["entry_indexes"],
            "planned_run_count": total,
            "finished_count": len(rows),
            "success_count": sum(bool(row.get("solved")) for row in rows),
            "router_request_count": sum(row.get("router_requests", 0) for row in rows),
            "qwen_route_count": sum(row.get("qwen_route_count", 0) for row in rows),
            "gpt_route_count": sum(row.get("gpt_route_count", 0) for row in rows),
            "route_error_count": sum(row.get("route_error_count", 0) for row in rows),
            "route_fallback_count": sum(
                row.get("route_fallback_count", 0) for row in rows
            ),
            "qwen_call_count": sum(row.get("qwen_calls", 0) for row in rows),
            "gpt_call_count": sum(row.get("gpt_calls", 0) for row in rows),
            "qwen_action_error_count": sum(
                row.get("qwen_action_error_count", 0) for row in rows
            ),
            "gpt_action_error_count": sum(
                row.get("gpt_action_error_count", 0) for row in rows
            ),
            "subprocess_timeout_count": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            **_averages(rows),
            "updated_at": utc_now(),
        },
    )


def runner_row(finished: int, total: int, result: dict[str, Any]) -> dict[str, Any]:
    """Build one JSONL progress record for ``runner.log``."""
    return {
        "finished": finished,
        "total": total,
        "task": result["task"],
        "entry_index": result["entry_index"],
        "seed": result["seed"],
        "success": bool(result.get("solved")),
        "router_requests": result.get("router_requests", 0),
        "qwen_routes": result.get("qwen_route_count", 0),
        "gpt_routes": result.get("gpt_route_count", 0),
        "route_errors": result.get("route_error_count", 0),
        "qwen_calls": result.get("qwen_calls", 0),
        "gpt_calls": result.get("gpt_calls", 0),
        "subprocess_timeout": bool(result.get("subprocess_timeout")),
        "elapsed_s": result.get("elapsed_s"),
    }


def append_runner_log(path: Path, row: dict[str, Any]) -> None:
    """Append one progress record and flush it for live monitoring."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def backfill_runner_log(path: Path, rows: list[dict[str, Any]], total: int) -> None:
    """Create a progress log from completed results when it is missing."""
    if path.exists() or not rows:
        return
    ordered = sorted(rows, key=lambda row: row.get("finished_at", ""))
    for index, row in enumerate(ordered, start=1):
        append_runner_log(path, runner_row(index, total, row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--qwen-memory-dir", type=Path, default=DEFAULT_QWEN_MEMORY)
    parser.add_argument(
        "--gpt-memory-source", type=Path, default=DEFAULT_GPT_MEMORY_SOURCE
    )
    parser.add_argument("--assets-path", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--entry-indexes", default="2,3")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--subprocess-timeout-s", type=int, default=1200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--worker-base-url", default="https://cloud.infini-ai.com/maas")
    parser.add_argument("--gpt-base-url", default="https://ai.soruxgpt.com")
    parser.add_argument("--gpt-model", default=GPT_MODEL)
    parser.add_argument("--gpt-reasoning-effort", default="high")
    parser.add_argument("--router-temperature", type=float, default=0.0)
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
    if not 0 <= args.router_temperature <= 2:
        parser.error("--router-temperature must be between 0 and 2")
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
        raise RuntimeError(f"Qwen router runner not found: {RUNNER}")
    if not args.assets_path.is_dir():
        raise RuntimeError(f"RoboTwin assets not found: {args.assets_path}")
    if not args.model_path.is_dir():
        raise RuntimeError(f"LingBot model not found: {args.model_path}")
    if not os.environ.get("INFINI_API_KEY"):
        raise RuntimeError("INFINI_API_KEY is required")
    if not os.environ.get("RPENT_API_KEY"):
        raise RuntimeError("RPENT_API_KEY is required")

    runs = load_runs(args.entry_indexes, args.task_limit)
    validate_memory(runs, args.qwen_memory_dir, args.gpt_memory_source)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    gpt_memory_files = prepare_gpt_memories(
        args.run_dir, args.gpt_memory_source, {run["task_name"] for run in runs}
    )

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise RuntimeError("--gpus must not be empty")
    settings = {
        "assets_path": str(args.assets_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "worker_base_url": args.worker_base_url,
        "gpt_base_url": args.gpt_base_url,
        "gpt_model": args.gpt_model,
        "gpt_reasoning_effort": args.gpt_reasoning_effort,
        "router_temperature": args.router_temperature,
        "timeout_s": args.timeout_s,
        "subprocess_timeout_s": args.subprocess_timeout_s,
        "max_turns": args.max_turns,
        "max_episode_steps": args.max_episode_steps,
        "entry_indexes": args.entry_indexes,
        "rpent": args.rpent,
        "vla_endpoints": {},
    }
    write_json(
        args.run_dir / "batch.json",
        {
            "stage": "RoboTwin per-turn Qwen routing between Qwen and GPT",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "router_model": WORKER_MODEL,
            "router_temperature": args.router_temperature,
            "gpt_model": args.gpt_model,
            "gpt_reasoning_effort": args.gpt_reasoning_effort,
            "entry_indexes": args.entry_indexes,
            "task_limit": args.task_limit,
            "qwen_memory_dir": str(args.qwen_memory_dir.resolve()),
            "gpt_memory_source": str(args.gpt_memory_source.resolve()),
            "timeout_s": args.timeout_s,
            "tool_timeout_s": args.timeout_s,
            "gpus": gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "planned_run_count": len(runs),
            "started_at": utc_now(),
        },
    )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/result.json"))
    ]
    finished = {(row["task"], row["entry_index"], row["seed"]) for row in rows}
    pending = [
        run
        for run in runs
        if (run["task_name"], run["entry_index"], run["seed"]) not in finished
    ]
    runner_log = args.run_dir / "runner.log"
    backfill_runner_log(runner_log, rows, len(runs))
    update_status(args.run_dir, rows, len(runs), settings)

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
                max_workers=args.workers_per_gpu, thread_name_prefix=f"gpu{gpu}"
            )
            executors.append(executor)
            for run in batch:
                future = executor.submit(
                    run_episode,
                    run,
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
            update_status(args.run_dir, rows, len(runs), settings)
            progress = runner_row(len(rows), len(runs), result)
            append_runner_log(runner_log, progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        for process, _ in shared_vlas.values():
            stop_shared_vla(process)

    update_status(args.run_dir, rows, len(runs), settings)
    write_json(
        args.run_dir / "summary.json",
        {
            "runs": len(runs),
            "success": sum(bool(row.get("solved")) for row in rows),
            "router_requests": sum(row.get("router_requests", 0) for row in rows),
            "qwen_routes": sum(row.get("qwen_route_count", 0) for row in rows),
            "gpt_routes": sum(row.get("gpt_route_count", 0) for row in rows),
            "route_errors": sum(row.get("route_error_count", 0) for row in rows),
            "route_fallbacks": sum(row.get("route_fallback_count", 0) for row in rows),
            "qwen_calls": sum(row.get("qwen_calls", 0) for row in rows),
            "gpt_calls": sum(row.get("gpt_calls", 0) for row in rows),
            "qwen_action_errors": sum(
                row.get("qwen_action_error_count", 0) for row in rows
            ),
            "gpt_action_errors": sum(
                row.get("gpt_action_error_count", 0) for row in rows
            ),
            "subprocess_timeouts": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            **_averages(rows),
            "finished_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
