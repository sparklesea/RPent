#!/usr/bin/env python3
"""Evaluate the RoboTwin BRACE-style event-triggered replanning baseline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "robots" / "robotwin" / "eval" / "demo_randomized.json"
RUNNER = Path(__file__).resolve().parent / "run_brace_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.8-27b"
SENIOR_MODEL = "openai:gpt-6-astra"
DEFAULT_CHECKIN = 8
DEFAULT_ERRORS = 4
DEFAULT_STALL_WINDOW = 4
DEFAULT_STALL_SPAN_M = 0.01
DEFAULT_MAX_VETOES = 1
DEFAULT_EPISODE_TIMEOUT_S = 1200
DEFAULT_CLEANUP_GRACE_S = 60
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
DEFAULT_RUN_DIR = ROOT / "runs" / "robotwin-brace-event-eval"
PHASE = "brace-event-eval"
POLICY = "gpt_plan_qwen_execute_brace_event_replan"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_candidate_verify_eval import validate_memory  # noqa: E402
from run_collab_eval import load_runs, parse_episode, prepare_gpt_memories  # noqa: E402
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
    write_json,
)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the runner and every child it started, then reap it."""
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
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def parse_senior_reasons(output_dir: Path) -> dict[str, int]:
    """Count senior turns by trigger reason from the escalation log."""
    path = output_dir / "escalation.jsonl"
    counts: Counter[str] = Counter()
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("role") == "senior":
            counts[str(row.get("reason") or "unknown")] += 1
    return dict(sorted(counts.items()))


def run_episode(
    run: dict[str, Any],
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
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if not existing.get("runner_error"):
            return existing

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
            "SESSION_TIMEOUT_S": str(settings["planner_timeout_s"]),
            "TOOL_TIMEOUT_S": str(settings["planner_timeout_s"]),
            "MAX_TURNS": str(settings["max_turns"]),
            "MAX_EPISODE_STEPS": str(settings["max_episode_steps"]),
            "RPENT_BRACE": "1",
            "RPENT_BRACE_CHECKIN": str(settings["checkin"]),
            "RPENT_BRACE_ERRORS": str(settings["errors"]),
            "RPENT_BRACE_STALL_WINDOW": str(settings["stall_window"]),
            "RPENT_BRACE_STALL_SPAN_M": str(settings["stall_span_m"]),
            "RPENT_BRACE_MAX_VETOES": str(settings["max_vetoes"]),
            "VLA_ENDPOINT": settings["vla_endpoints"][gpu],
        }
    )

    started = time.monotonic()
    timed_out = False
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
        process = None
        try:
            process = subprocess.Popen(
                [str(RUNNER)],
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return_code = process.wait(timeout=settings["timeout_s"])
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
            stdout.write(
                f"\n[runner] subprocess timeout after "
                f"{settings['timeout_s']}s\n"
            )
            if process is not None:
                _terminate_process_group(process)

    parsed = parse_episode(output_dir)
    task_success = parsed.get("environment_success") is True
    clean_exit = return_code == 0 and not timed_out
    finish_success = parsed.get("finish_status") == "success"

    result = {
        "phase": PHASE,
        "task": task_name,
        "entry_index": entry_index,
        "seed": seed,
        "task_language": run["task_language"],
        "gpu": gpu,
        "worker_model": WORKER_MODEL,
        "worker_reasoning_effort": "none",
        "senior_model": SENIOR_MODEL,
        "senior_reasoning_effort": "high",
        "policy": POLICY,
        "triggers": [
            "initial_plan",
            "tool_errors",
            "eef_stall",
            "periodic_checkin",
        ],
        "brace_checkin": settings["checkin"],
        "brace_errors": settings["errors"],
        "brace_stall_window": settings["stall_window"],
        "brace_stall_span_m": settings["stall_span_m"],
        "brace_max_vetoes": settings["max_vetoes"],
        "qwen_memory_dir": str(qwen_memory_dir.resolve()),
        "gpt_memory_file": str(gpt_memory_file.resolve()),
        "episode_timeout_s": settings["timeout_s"],
        "planner_timeout_s": settings["planner_timeout_s"],
        "tool_timeout_s": settings["planner_timeout_s"],
        "exit_code": return_code,
        "subprocess_timeout": timed_out,
        "clean_exit": clean_exit,
        "task_success": task_success,
        "clean_success": task_success and clean_exit and finish_success,
        "elapsed_s": round(time.monotonic() - started, 1),
        **parsed,
        "senior_reason_counts": parse_senior_reasons(output_dir),
        "finished_at": utc_now(),
    }
    write_json(result_path, result)
    return result


def _averages(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("solved")]
    clean_successful = [row for row in rows if row.get("clean_success")]
    count = len(successful)
    clean_count = len(clean_successful)

    def average(
        values: list[dict[str, Any]], key: str, *, digits: int = 2
    ) -> float | None:
        if not values:
            return None
        return round(sum(row.get(key, 0) or 0 for row in values) / len(values), digits)

    return {
        "success_count": count,
        "clean_success_count": clean_count,
        "avg_gpt_calls_per_success": average(successful, "gpt_calls"),
        "avg_turns_per_success": average(successful, "turns"),
        "avg_elapsed_s_per_success": average(successful, "elapsed_s", digits=1),
        "avg_gpt_calls_per_clean_success": average(clean_successful, "gpt_calls"),
        "avg_elapsed_s_per_clean_success": average(
            clean_successful, "elapsed_s", digits=1
        ),
    }


def update_status(
    run_dir: Path,
    rows: list[dict[str, Any]],
    total: int,
    entry_indexes: list[int],
    settings: dict[str, Any],
) -> None:
    by_entry = {
        entry_index: {
            "finished": sum(row.get("entry_index") == entry_index for row in rows),
            "success": sum(
                row.get("entry_index") == entry_index and bool(row.get("solved"))
                for row in rows
            ),
            "clean_success": sum(
                row.get("entry_index") == entry_index
                and bool(row.get("clean_success"))
                for row in rows
            ),
            "gpt_calls": sum(
                row.get("entry_index") == entry_index and row.get("gpt_calls", 0)
                for row in rows
            ),
        }
        for entry_index in entry_indexes
    }
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row.get("senior_reason_counts") or {})
    write_json(
        run_dir / "status.json",
        {
            "phase": PHASE,
            "policy": POLICY,
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "brace_checkin": settings["checkin"],
            "brace_errors": settings["errors"],
            "brace_stall_window": settings["stall_window"],
            "brace_stall_span_m": settings["stall_span_m"],
            "task_config": "demo_randomized",
            "entry_indexes": entry_indexes,
            "planned_run_count": total,
            "finished_count": len(rows),
            "success_count": sum(bool(row.get("solved")) for row in rows),
            "clean_success_count": sum(
                bool(row.get("clean_success")) for row in rows
            ),
            "gpt_call_count": sum(row.get("gpt_calls", 0) for row in rows),
            "senior_reason_counts": dict(sorted(reason_counts.items())),
            "subprocess_timeout_count": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            "runner_error_count": sum(bool(row.get("runner_error")) for row in rows),
            "by_entry_index": by_entry,
            **_averages(rows),
            "updated_at": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN_DIR)
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
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_EPISODE_TIMEOUT_S)
    parser.add_argument("--cleanup-grace-s", type=int, default=DEFAULT_CLEANUP_GRACE_S)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--checkin", type=int, default=DEFAULT_CHECKIN)
    parser.add_argument("--errors", type=int, default=DEFAULT_ERRORS)
    parser.add_argument("--stall-window", type=int, default=DEFAULT_STALL_WINDOW)
    parser.add_argument("--stall-span-m", type=float, default=DEFAULT_STALL_SPAN_M)
    parser.add_argument("--max-vetoes", type=int, default=DEFAULT_MAX_VETOES)
    parser.add_argument(
        "--worker-base-url", default="https://cloud.infini-ai.com/maas"
    )
    parser.add_argument("--senior-base-url", default="https://ai.soruxgpt.com")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.timeout_s < 1 or args.cleanup_grace_s < 0:
        parser.error("--timeout-s must be positive and --cleanup-grace-s non-negative")
    if args.cleanup_grace_s >= args.timeout_s:
        parser.error("--cleanup-grace-s must be smaller than --timeout-s")
    if args.max_turns < 1 or args.max_episode_steps < 1:
        parser.error("--max-turns and --max-episode-steps must be positive")
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("--task-limit must be positive")
    if args.checkin < 1 or args.errors < 1:
        parser.error("--checkin and --errors must be positive")
    if args.stall_window < 2 or args.stall_span_m < 0:
        parser.error("--stall-window must be >= 2 and --stall-span-m non-negative")
    if args.max_vetoes < 0:
        parser.error("--max-vetoes must not be negative")
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
        raise RuntimeError(f"evaluation runner not found: {RUNNER}")
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

    planner_timeout_s = args.timeout_s - args.cleanup_grace_s
    settings = {
        "assets_path": str(args.assets_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "worker_base_url": args.worker_base_url,
        "senior_base_url": args.senior_base_url,
        "timeout_s": args.timeout_s,
        "planner_timeout_s": planner_timeout_s,
        "cleanup_grace_s": args.cleanup_grace_s,
        "max_turns": args.max_turns,
        "max_episode_steps": args.max_episode_steps,
        "checkin": args.checkin,
        "errors": args.errors,
        "stall_window": args.stall_window,
        "stall_span_m": args.stall_span_m,
        "max_vetoes": args.max_vetoes,
        "workers_per_gpu": args.workers_per_gpu,
        "vla_endpoints": {},
    }
    write_json(
        args.run_dir / "batch.json",
        {
            "stage": "RoboTwin BRACE-style event-triggered replanning baseline",
            "policy": POLICY,
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "triggers": [
                "initial_plan",
                "tool_errors",
                "eef_stall",
                "periodic_checkin",
            ],
            "brace_checkin": args.checkin,
            "brace_errors": args.errors,
            "brace_stall_window": args.stall_window,
            "brace_stall_span_m": args.stall_span_m,
            "brace_max_vetoes": args.max_vetoes,
            "entry_indexes": args.entry_indexes,
            "task_limit": args.task_limit,
            "qwen_memory_dir": str(args.qwen_memory_dir.resolve()),
            "gpt_memory_source": str(args.gpt_memory_source.resolve()),
            "timeout_s": args.timeout_s,
            "planner_timeout_s": planner_timeout_s,
            "tool_timeout_s": planner_timeout_s,
            "cleanup_grace_s": args.cleanup_grace_s,
            "gpus": [value.strip() for value in args.gpus.split(",") if value.strip()],
            "workers_per_gpu": args.workers_per_gpu,
            "planned_run_count": len(runs),
            "started_at": utc_now(),
        },
    )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/result.json"))
    ]
    rows = [row for row in rows if not row.get("runner_error")]
    finished_runs = {
        (row["task"], row["entry_index"], row["seed"]) for row in rows
    }
    pending = [
        run
        for run in runs
        if (run["task_name"], run["entry_index"], run["seed"]) not in finished_runs
    ]
    update_status(args.run_dir, rows, len(runs), args.entry_indexes, settings)

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    batches = {gpu: pending[index::len(gpus)] for index, gpu in enumerate(gpus)}
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
                started = time.monotonic()
                future = executor.submit(
                    run_episode,
                    run,
                    run_dir=args.run_dir,
                    qwen_memory_dir=args.qwen_memory_dir,
                    gpt_memory_file=gpt_memory_files[run["task_name"]],
                    gpu=gpu,
                    settings=settings,
                )
                futures[future] = (run, gpu, started)

        for future in as_completed(futures):
            run, gpu, started = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "phase": PHASE,
                    "task": run["task_name"],
                    "entry_index": run["entry_index"],
                    "seed": run["seed"],
                    "task_language": run["task_language"],
                    "gpu": gpu,
                    "worker_model": WORKER_MODEL,
                    "worker_reasoning_effort": "none",
                    "senior_model": SENIOR_MODEL,
                    "senior_reasoning_effort": "high",
                    "policy": POLICY,
                    "brace_checkin": settings["checkin"],
                    "brace_errors": settings["errors"],
                    "brace_stall_window": settings["stall_window"],
                    "brace_stall_span_m": settings["stall_span_m"],
                    "brace_max_vetoes": settings["max_vetoes"],
                    "qwen_memory_dir": str(args.qwen_memory_dir.resolve()),
                    "gpt_memory_file": str(
                        gpt_memory_files[run["task_name"]].resolve()
                    ),
                    "episode_timeout_s": settings["timeout_s"],
                    "planner_timeout_s": settings["planner_timeout_s"],
                    "tool_timeout_s": settings["planner_timeout_s"],
                    "exit_code": None,
                    "subprocess_timeout": False,
                    "clean_exit": False,
                    "solved": False,
                    "environment_success": None,
                    "task_success": False,
                    "clean_success": False,
                    "gpt_calls": 0,
                    "turns": None,
                    "finish_status": None,
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "runner_error": f"{type(error).__name__}: {error}",
                    "finished_at": utc_now(),
                }
                result_path = (
                    args.run_dir
                    / f"{run['task_name']}_entry{run['entry_index']}_s{run['seed']}"
                    / "result.json"
                )
                try:
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(result_path, result)
                except OSError as write_error:
                    print(
                        json.dumps(
                            {
                                "task": result["task"],
                                "entry_index": result["entry_index"],
                                "seed": result["seed"],
                                "result_write_error": (
                                    f"{type(write_error).__name__}: {write_error}"
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            rows.append(result)
            update_status(args.run_dir, rows, len(runs), args.entry_indexes, settings)
            print(
                json.dumps(
                    {
                        "finished": len(rows),
                        "total": len(runs),
                        "task": result["task"],
                        "entry_index": result["entry_index"],
                        "seed": result["seed"],
                        "success": bool(result["solved"]),
                        "clean_success": bool(result.get("clean_success")),
                        "finish_status": result.get("finish_status"),
                        "gpt_calls": result.get("gpt_calls", 0),
                        "senior_reasons": result.get("senior_reason_counts", {}),
                        "subprocess_timeout": bool(result.get("subprocess_timeout")),
                        "runner_error": result.get("runner_error"),
                        "elapsed_s": result.get("elapsed_s"),
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

    update_status(args.run_dir, rows, len(runs), args.entry_indexes, settings)
    write_json(
        args.run_dir / "summary.json",
        {
            "runs": len(runs),
            "success": sum(bool(row.get("solved")) for row in rows),
            "clean_success": sum(bool(row.get("clean_success")) for row in rows),
            "gpt_calls": sum(row.get("gpt_calls", 0) for row in rows),
            "subprocess_timeout_count": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            **_averages(rows),
            "finished_at": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
