#!/usr/bin/env python3
"""Evaluate the RoboTwin GPT-plan / Qwen-execute timed-replan baseline.

The first planning turn is GPT-6.  Qwen handles every other turn.  If the episode
is still running after 600 s, GPT-6 replans once at that midpoint.  Qwen and GPT
read separate task memories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "robots" / "robotwin" / "eval" / "demo_randomized.json"
RUNNER = Path(__file__).resolve().parent / "run_plan_timed_episode.sh"
WORKER_MODEL = "openai-chat:qwen3.8-27b"
SENIOR_MODEL = "openai:gpt-6-astra"
PLAN_TIME_REPLAN_S = 600.0
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
DEFAULT_RUN_DIR = ROOT / "runs" / "robotwin-plan-timed-eval"
PHASE = "plan-timed-eval"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_candidate_verify_eval import validate_memory  # noqa: E402
from run_collab_eval import load_runs, parse_episode, prepare_gpt_memories  # noqa: E402
from run_ecc_stage1 import (  # noqa: E402
    start_shared_vla,
    stop_shared_vla,
    utc_now,
    write_json,
)


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
            "RPENT_PLAN_TIMED": "1",
            "RPENT_PLAN_TIME_REPLAN_S": str(settings["replan_interval_s"]),
            "VLA_ENDPOINT": settings["vla_endpoints"][gpu],
        }
    )

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
        "senior_model": SENIOR_MODEL,
        "senior_reasoning_effort": "high",
        "policy": "gpt_plan_qwen_execute_timed_replan",
        "replan_interval_s": settings["replan_interval_s"],
        "qwen_memory_dir": str(qwen_memory_dir.resolve()),
        "gpt_memory_file": str(gpt_memory_file.resolve()),
        "planner_timeout_s": settings["timeout_s"],
        "tool_timeout_s": settings["timeout_s"],
        "exit_code": return_code,
        "subprocess_timeout": timed_out,
        "elapsed_s": round(time.monotonic() - started, 1),
        **parse_episode(output_dir),
        "senior_reason_counts": parse_senior_reasons(output_dir),
        "finished_at": utc_now(),
    }
    write_json(result_path, result)
    return result


def update_status(
    run_dir: Path,
    rows: list[dict[str, Any]],
    total: int,
    entry_indexes: list[int],
    replan_interval_s: float = PLAN_TIME_REPLAN_S,
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
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row.get("senior_reason_counts") or {})
    write_json(
        run_dir / "status.json",
        {
            "phase": PHASE,
            "policy": "gpt_plan_qwen_execute_timed_replan",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "replan_interval_s": replan_interval_s,
            "task_config": "demo_randomized",
            "entry_indexes": entry_indexes,
            "planned_run_count": total,
            "finished_count": len(rows),
            "success_count": sum(bool(row.get("solved")) for row in rows),
            "gpt_call_count": sum(row.get("gpt_calls", 0) for row in rows),
            "senior_reason_counts": dict(sorted(reason_counts.items())),
            "subprocess_timeout_count": sum(
                bool(row.get("subprocess_timeout")) for row in rows
            ),
            "by_entry_index": by_entry,
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
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--subprocess-timeout-s", type=int, default=1200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--replan-interval-s", type=float, default=PLAN_TIME_REPLAN_S)
    parser.add_argument(
        "--worker-base-url", default="https://cloud.infini-ai.com/maas"
    )
    parser.add_argument("--senior-base-url", default="https://ai.soruxgpt.com")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.timeout_s < 1 or args.subprocess_timeout_s < 1:
        parser.error("timeouts must be positive")
    if args.max_turns < 1 or args.max_episode_steps < 1:
        parser.error("--max-turns and --max-episode-steps must be positive")
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("--task-limit must be positive")
    if args.replan_interval_s < 0:
        parser.error("--replan-interval-s must not be negative")
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

    settings = {
        "assets_path": str(args.assets_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "worker_base_url": args.worker_base_url,
        "senior_base_url": args.senior_base_url,
        "timeout_s": args.timeout_s,
        "subprocess_timeout_s": args.subprocess_timeout_s,
        "max_turns": args.max_turns,
        "max_episode_steps": args.max_episode_steps,
        "replan_interval_s": args.replan_interval_s,
        "workers_per_gpu": args.workers_per_gpu,
        "vla_endpoints": {},
    }
    write_json(
        args.run_dir / "batch.json",
        {
            "stage": "RoboTwin GPT-plan / Qwen-execute timed-replan baseline",
            "policy": "gpt_plan_qwen_execute_timed_replan",
            "worker_model": WORKER_MODEL,
            "worker_reasoning_effort": "none",
            "senior_model": SENIOR_MODEL,
            "senior_reasoning_effort": "high",
            "replan_interval_s": args.replan_interval_s,
            "entry_indexes": args.entry_indexes,
            "task_limit": args.task_limit,
            "qwen_memory_dir": str(args.qwen_memory_dir.resolve()),
            "gpt_memory_source": str(args.gpt_memory_source.resolve()),
            "timeout_s": args.timeout_s,
            "tool_timeout_s": args.timeout_s,
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
    finished_runs = {
        (row["task"], row["entry_index"], row["seed"]) for row in rows
    }
    pending = [
        run
        for run in runs
        if (run["task_name"], run["entry_index"], run["seed"]) not in finished_runs
    ]
    update_status(
        args.run_dir, rows, len(runs), args.entry_indexes, args.replan_interval_s
    )

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
            update_status(
                args.run_dir,
                rows,
                len(runs),
                args.entry_indexes,
                args.replan_interval_s,
            )
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
                        "senior_reasons": result.get("senior_reason_counts", {}),
                        "subprocess_timeout": bool(result.get("subprocess_timeout")),
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

    update_status(
        args.run_dir, rows, len(runs), args.entry_indexes, args.replan_interval_s
    )
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
    raise SystemExit(main())
