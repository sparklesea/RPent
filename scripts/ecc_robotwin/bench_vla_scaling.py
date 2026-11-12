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

"""Benchmark LingBot-VLA serving under a varying number of client workers.

The benchmark starts one shared VLA server on a single GPU, then drives it
with independent ``WebsocketClientPolicy`` connections. Every worker sends a
fixed number of sequential requests, so a level with ``N`` workers measures
how the server behaves when ``N`` RoboTwin-style clients share one VLA.

The server records ``server_timing.infer_ms`` from just before acquiring its
global inference lock. Consequently that value includes FIFO waiting when
requests overlap. This script:

* records client latency and ``infer_ms`` for each request;
* subtracts the single-worker ``infer_ms`` baseline to estimate queueing;
* samples GPU utilization, device memory, and the VLA process memory; and
* writes per-request, per-level, and aggregate summaries.

The client side is deliberately VLA-only: it does not start RoboTwin envs.
Worker growth therefore shows server saturation and FIFO behavior, not the
GPU or RAM cost of additional simulator environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from run_ecc_stage1 import start_shared_vla, stop_shared_vla  # noqa: E402

DEFAULT_MODEL_PATH = (
    ROOT
    / "runs"
    / "robotwin-ecc-stage1"
    / "models"
    / "LingBot-VLA-RoboTwin-EEF-ckpt1500"
)
DEFAULT_TASK_LANGUAGE = "Lift the hammer for nails and hit the block"
DEFAULT_STATE = np.asarray(
    [
        -0.30044541,
        -0.31786108,
        0.94221973,
        0.70300003,
        0.00827169,
        -0.00377289,
        0.71113171,
        0.0,
        -0.01035806,
        -0.08328216,
        0.91135000,
        0.70300003,
        0.00827169,
        -0.00377289,
        0.71113171,
        0.0,
    ],
    dtype=np.float32,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def load_images(
    image_dir: Path | None,
    height: int,
    width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load three RoboTwin views, or create deterministic synthetic inputs."""
    if image_dir is not None:
        names = ("head_rgb.png", "left_wrist_rgb.png", "right_wrist_rgb.png")
        images: list[np.ndarray] = []
        for name in names:
            path = image_dir / name / "00.png"
            if not path.is_file():
                raise FileNotFoundError(f"missing benchmark image: {path}")
            from PIL import Image

            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if len({image.shape for image in images}) != 1:
            raise ValueError("all benchmark images must have the same shape")
        return images[0], images[1], images[2]

    rng = np.random.default_rng(seed)
    shape = (height, width, 3)
    return tuple(rng.integers(0, 256, size=shape, dtype=np.uint8) for _ in range(3))


def make_payload(
    head: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    task_language: str,
) -> dict[str, Any]:
    """Build the same observation contract used by ``LingBotVLAClient``."""
    return {
        "observation.images.cam_high": head,
        "observation.images.cam_left_wrist": left_wrist,
        "observation.images.cam_right_wrist": right_wrist,
        "observation.state": DEFAULT_STATE.copy(),
        "task": task_language,
    }


def endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """Return the host and port from a ``ws://`` endpoint."""
    parsed = urlparse(endpoint)
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"invalid WebSocket endpoint: {endpoint!r}")
    return parsed.hostname, int(parsed.port)


def close_client(client: WebsocketClientPolicy) -> None:
    """Best-effort close for the official websocket client."""
    connection = getattr(client, "_ws", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def percentile(values: list[float], q: float) -> float | None:
    """Return a percentile, or ``None`` for an empty sample."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: list[float]) -> float | None:
    """Return a mean, or ``None`` for an empty sample."""
    return float(statistics.fmean(values)) if values else None


def optional_float(value: Any) -> float | None:
    """Convert a possibly missing numeric value to ``float``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def gpu_process_memory_mb(handle: Any, pid: int) -> float | None:
    """Return the compute memory attributed to ``pid`` when NVML exposes it."""
    pynvml = sys.modules.get("pynvml")
    if pynvml is None:
        return None
    getters = (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    )
    for getter_name in getters:
        getter = getattr(pynvml, getter_name, None)
        if getter is None:
            continue
        try:
            processes = getter(handle)
        except Exception:
            continue
        for process in processes:
            if int(getattr(process, "pid", -1)) != pid:
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used is None:
                return None
            return float(used) / (1024.0 * 1024.0)
    return None


class GpuSampler(threading.Thread):
    """Continuously sample one GPU and the benchmark's VLA process."""

    def __init__(self, gpu_index: int, server_pid: int, interval_s: float):
        super().__init__(name="gpu-sampler", daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._gpu_index = gpu_index
        self._server_pid = server_pid
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {
            "phase": "startup",
            "workers": None,
            "repeat": None,
        }

    def set_context(
        self,
        *,
        phase: str,
        workers: int | None = None,
        repeat: int | None = None,
    ) -> None:
        """Tag subsequent samples with the currently running benchmark phase."""
        with self._lock:
            self._context = {
                "phase": phase,
                "workers": workers,
                "repeat": repeat,
            }

    def sample_once(self, phase: str) -> dict[str, Any]:
        """Take and retain one sample immediately."""
        pynvml = self._pynvml
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        sample = {
            "timestamp": utc_now(),
            "phase": phase,
            "gpu_index": self._gpu_index,
            "gpu_util_pct": float(utilization.gpu),
            "gpu_memory_used_mb": float(memory.used) / (1024.0 * 1024.0),
            "gpu_memory_total_mb": float(memory.total) / (1024.0 * 1024.0),
            "server_process_memory_mb": gpu_process_memory_mb(
                self._handle, self._server_pid
            ),
        }
        with self._lock:
            sample.update(self._context)
            self._samples.append(sample)
        return sample

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of all retained samples."""
        with self._lock:
            return [dict(sample) for sample in self._samples]

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self.sample_once("sampling")
            except Exception:
                continue

    def stop(self) -> list[dict[str, Any]]:
        """Stop sampling and return a snapshot of all retained samples."""
        self._stop_event.set()
        self.join(timeout=5.0)
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
        return samples


class LiveLog:
    """Print progress and mirror it to ``runner.log``."""

    def __init__(self, path: Path):
        self._path = path

    def write(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def run_wave(
    *,
    workers: int,
    requests_per_worker: int,
    warmup_per_worker: int,
    endpoint: str,
    payload: dict[str, Any],
    repeat: int,
    barrier_timeout_s: float,
    request_period_s: float | None = None,
) -> dict[str, Any]:
    """Run one wave of independent client workers.

    With ``request_period_s=None`` every worker sends continuously, which is
    the saturation/stress test. With a period, workers are evenly phase-shifted
    and each sends one request per period, approximating real agent duty cycles.
    """
    host, port = endpoint_host_port(endpoint)
    connect_barrier = threading.Barrier(workers + 1)
    ready_barrier = threading.Barrier(workers + 1)
    start_event = threading.Event()
    errors: list[str] = []
    errors_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    worker_end_ns: list[int] = []
    schedule_start_ns = [0]

    def worker(worker_id: int) -> None:
        client: WebsocketClientPolicy | None = None
        try:
            connect_started = time.perf_counter()
            client = WebsocketClientPolicy(host=host, port=port)
            connect_ms = (time.perf_counter() - connect_started) * 1000.0
            connect_barrier.wait(timeout=barrier_timeout_s)

            for _ in range(warmup_per_worker):
                client.infer(payload)

            ready_barrier.wait(timeout=barrier_timeout_s)
            while not start_event.wait(timeout=0.25):
                if errors:
                    return
            local_records: list[dict[str, Any]] = []
            phase_s = (
                (worker_id / workers) * request_period_s
                if request_period_s is not None
                else 0.0
            )
            for sequence in range(requests_per_worker):
                scheduled_s = phase_s + sequence * (request_period_s or 0.0)
                if request_period_s is not None:
                    target_ns = schedule_start_ns[0] + int(scheduled_s * 1e9)
                    while True:
                        remaining_s = (target_ns - time.perf_counter_ns()) / 1e9
                        if remaining_s <= 0:
                            break
                        time.sleep(min(0.5, remaining_s))
                request_start_ns = time.perf_counter_ns()
                response = client.infer(payload)
                request_end_ns = time.perf_counter_ns()
                client_latency_ms = (request_end_ns - request_start_ns) / 1e6
                timing = response.get("server_timing") or {}
                infer_ms = optional_float(timing.get("infer_ms"))
                prev_total_ms = optional_float(timing.get("prev_total_ms"))
                action = response.get("action")
                action_shape = list(action.shape) if hasattr(action, "shape") else None
                local_records.append(
                    {
                        "workers": workers,
                        "repeat": repeat,
                        "worker_id": worker_id,
                        "sequence": sequence,
                        "scheduled_s": scheduled_s,
                        "connect_ms": connect_ms,
                        "client_latency_ms": client_latency_ms,
                        "infer_ms": infer_ms,
                        "client_overhead_ms": (
                            client_latency_ms - infer_ms
                            if infer_ms is not None
                            else None
                        ),
                        "prev_total_ms": prev_total_ms,
                        "action_shape": action_shape,
                        "request_start_unix_s": time.time()
                        - (time.perf_counter() - request_start_ns / 1e9),
                    }
                )
            worker_end_ns.append(time.perf_counter_ns())
            with records_lock:
                records.extend(local_records)
        except BaseException as exc:
            message = f"worker {worker_id}: {type(exc).__name__}: {exc}"
            with errors_lock:
                errors.append(message)
            connect_barrier.abort()
            ready_barrier.abort()
            start_event.set()
        finally:
            if client is not None:
                close_client(client)

    threads = [
        threading.Thread(target=worker, args=(worker_id,), name=f"client-{worker_id}")
        for worker_id in range(workers)
    ]
    for thread in threads:
        thread.start()
    try:
        connect_barrier.wait(timeout=barrier_timeout_s)
        ready_barrier.wait(timeout=barrier_timeout_s)
    except threading.BrokenBarrierError as exc:
        errors.append(f"barrier failed: {exc}")
        start_event.set()

    start_ns = time.perf_counter_ns()
    schedule_start_ns[0] = start_ns
    start_event.set()
    for thread in threads:
        thread.join()
    finish_ns = max(worker_end_ns) if worker_end_ns else time.perf_counter_ns()
    if errors:
        raise RuntimeError("; ".join(errors))

    wall_s = (finish_ns - start_ns) / 1e9
    expected = workers * requests_per_worker
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} requests, received {len(records)}")
    return {
        "workers": workers,
        "repeat": repeat,
        "requests": len(records),
        "wall_s": wall_s,
        "throughput_rps": len(records) / wall_s if wall_s > 0 else None,
        "request_period_s": request_period_s,
        "records": records,
    }


def aggregate_level(
    *,
    workers: int,
    waves: list[dict[str, Any]],
    baseline_infer_ms: float | None,
    gpu_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate latency, throughput, and GPU samples for one worker level."""
    records = [record for wave in waves for record in wave["records"]]
    client_latencies = [
        float(record["client_latency_ms"])
        for record in records
        if record["client_latency_ms"] is not None
    ]
    infer_values = [
        float(record["infer_ms"])
        for record in records
        if record["infer_ms"] is not None
    ]
    overheads = [
        float(record["client_overhead_ms"])
        for record in records
        if record["client_overhead_ms"] is not None
    ]
    queue_estimates = (
        [max(0.0, value - baseline_infer_ms) for value in infer_values]
        if baseline_infer_ms is not None
        else []
    )
    throughputs = [float(wave["throughput_rps"]) for wave in waves]
    gpu_utils = [
        float(sample["gpu_util_pct"])
        for sample in gpu_samples
        if sample.get("gpu_util_pct") is not None
    ]
    gpu_memory = [
        float(sample["gpu_memory_used_mb"])
        for sample in gpu_samples
        if sample.get("gpu_memory_used_mb") is not None
    ]
    server_memory = [
        float(sample["server_process_memory_mb"])
        for sample in gpu_samples
        if sample.get("server_process_memory_mb") is not None
    ]
    return {
        "workers": workers,
        "repeats": len(waves),
        "requests": len(records),
        "wall_s_mean": mean_or_none([float(wave["wall_s"]) for wave in waves]),
        "throughput_rps_mean": mean_or_none(throughputs),
        "throughput_rps_std": (
            float(statistics.pstdev(throughputs)) if len(throughputs) > 1 else 0.0
        ),
        "client_latency_ms_p50": percentile(client_latencies, 50),
        "client_latency_ms_p95": percentile(client_latencies, 95),
        "client_latency_ms_p99": percentile(client_latencies, 99),
        "infer_ms_p50": percentile(infer_values, 50),
        "infer_ms_p95": percentile(infer_values, 95),
        "infer_ms_p99": percentile(infer_values, 99),
        "queue_estimate_ms_p50": percentile(queue_estimates, 50),
        "queue_estimate_ms_p95": percentile(queue_estimates, 95),
        "queue_estimate_ms_max": max(queue_estimates) if queue_estimates else None,
        "client_overhead_ms_p50": percentile(overheads, 50),
        "client_overhead_ms_p95": percentile(overheads, 95),
        "gpu_util_pct_mean": mean_or_none(gpu_utils),
        "gpu_util_pct_p95": percentile(gpu_utils, 95),
        "gpu_memory_used_mb_mean": mean_or_none(gpu_memory),
        "gpu_memory_used_mb_p95": percentile(gpu_memory, 95),
        "server_process_memory_mb_mean": mean_or_none(server_memory),
        "server_process_memory_mb_p95": percentile(server_memory, 95),
        "waves": [
            {key: value for key, value in wave.items() if key != "records"}
            for wave in waves
        ],
    }


def write_json(path: Path, value: Any) -> None:
    """Write indented JSON atomically enough for a benchmark artifact."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the most useful level metrics as CSV."""
    columns = [
        "workers",
        "repeats",
        "requests",
        "throughput_rps_mean",
        "throughput_rps_std",
        "client_latency_ms_p50",
        "client_latency_ms_p95",
        "client_latency_ms_p99",
        "infer_ms_p50",
        "infer_ms_p95",
        "infer_ms_p99",
        "queue_estimate_ms_p50",
        "queue_estimate_ms_p95",
        "queue_estimate_ms_max",
        "client_overhead_ms_p50",
        "client_overhead_ms_p95",
        "gpu_util_pct_mean",
        "gpu_util_pct_p95",
        "gpu_memory_used_mb_mean",
        "gpu_memory_used_mb_p95",
        "server_process_memory_mb_mean",
        "server_process_memory_mb_p95",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact human-readable comparison table."""

    def number(value: Any, digits: int = 2) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# VLA-only worker scaling",
        "",
        "| workers | throughput req/s | client p95 ms | infer p95 ms | queue est. p95 ms | GPU util mean % | GPU mem mean MiB | server proc. mean MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["workers"]),
                    number(row.get("throughput_rps_mean"), 3),
                    number(row.get("client_latency_ms_p95")),
                    number(row.get("infer_ms_p95")),
                    number(row.get("queue_estimate_ms_p95")),
                    number(row.get("gpu_util_pct_mean")),
                    number(row.get("gpu_memory_used_mb_mean")),
                    number(row.get("server_process_memory_mb_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`queue est.` is `infer_ms` minus the worker=1 `infer_ms` median; "
            "the server timer starts before its inference lock, so it includes FIFO wait.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="physical GPU index")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to runs/robotwin-vla-scaling-<time>",
    )
    parser.add_argument(
        "--workers",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16,24"),
        help="comma-separated client counts",
    )
    parser.add_argument("--requests-per-worker", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("saturated", "periodic"),
        default="saturated",
        help=(
            "saturated sends as fast as possible; periodic spaces each worker's "
            "requests by --request-period-s"
        ),
    )
    parser.add_argument(
        "--request-period-s",
        type=float,
        default=75.0,
        help="per-worker request period in periodic mode (default: observed 75s)",
    )
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--barrier-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="directory containing head_rgb.png/00.png and the two wrist views",
    )
    parser.add_argument(
        "--image-height", type=int, default=240, help="synthetic image height"
    )
    parser.add_argument(
        "--image-width", type=int, default=320, help="synthetic image width"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-language", default=DEFAULT_TASK_LANGUAGE)
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="allow writing into a non-empty run directory",
    )
    args = parser.parse_args()
    if args.requests_per_worker < 1:
        parser.error("--requests-per-worker must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.sample_interval_s <= 0:
        parser.error("--sample-interval-s must be positive")
    if args.request_period_s <= 0:
        parser.error("--request-period-s must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"LingBot model not found: {args.model_path}")
    if not Path("/proc").is_dir():
        raise RuntimeError("/proc is required for process discovery")

    run_dir = args.run_dir or (
        ROOT / "runs" / f"robotwin-vla-scaling-{datetime.now():%Y%m%d-%H%M%S}"
    )
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; pass --allow-existing to reuse it"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    live = LiveLog(run_dir / "runner.log")

    current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    no_proxy = {item.strip() for item in current.split(",") if item.strip()}
    no_proxy.update({"127.0.0.1", "localhost", "::1"})
    os.environ["NO_PROXY"] = ",".join(sorted(no_proxy))

    head, left_wrist, right_wrist = load_images(
        args.image_dir, args.image_height, args.image_width, args.seed
    )
    payload = make_payload(head, left_wrist, right_wrist, args.task_language)
    batch = {
        "mode": "vla_only_worker_scaling",
        "load_mode": args.mode,
        "request_period_s": (
            args.request_period_s if args.mode == "periodic" else None
        ),
        "started_at": utc_now(),
        "gpu": args.gpu,
        "model_path": str(args.model_path.resolve()),
        "run_dir": str(run_dir),
        "workers": args.workers,
        "requests_per_worker": args.requests_per_worker,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "sample_interval_s": args.sample_interval_s,
        "image_source": str(args.image_dir.resolve())
        if args.image_dir
        else "synthetic",
        "image_shape": list(head.shape),
        "state_shape": list(DEFAULT_STATE.shape),
        "task_language": args.task_language,
        "server_timer_note": (
            "server_timing.infer_ms starts before the server's inference lock, "
            "so it includes FIFO wait; queue estimates subtract the worker=1 median"
        ),
    }
    write_json(run_dir / "batch.json", batch)
    live.write(f"benchmark started: {run_dir}")
    live.write(
        f"GPU={args.gpu} workers={args.workers} "
        f"requests/worker={args.requests_per_worker} repeats={args.repeats} "
        f"mode={args.mode}"
    )

    server: subprocess.Popen | None = None
    sampler: GpuSampler | None = None
    level_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    try:
        pre_server = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu))
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pre_server = {
                "gpu_memory_used_mb": float(memory.used) / (1024.0 * 1024.0),
                "gpu_memory_total_mb": float(memory.total) / (1024.0 * 1024.0),
            }
            pynvml.nvmlShutdown()
        except Exception as exc:
            live.write(
                f"NVML pre-server sample unavailable: {type(exc).__name__}: {exc}"
            )

        live.write("starting shared VLA server (model load can take several minutes)")
        startup_started = time.perf_counter()
        server, endpoint = start_shared_vla(
            args.gpu, args.model_path.resolve(), run_dir
        )
        batch["endpoint"] = endpoint
        batch["server_pid"] = server.pid
        batch["server_startup_s"] = time.perf_counter() - startup_started
        live.write(
            f"shared VLA ready at {endpoint} in {batch['server_startup_s']:.1f}s "
            f"pid={server.pid}"
        )

        sampler = GpuSampler(int(args.gpu), server.pid, args.sample_interval_s)
        sampler.start()
        sampler.set_context(phase="loaded")
        time.sleep(max(1.0, args.sample_interval_s * 5))
        loaded_sample = sampler.sample_once("loaded")
        batch["gpu_memory_pre_server_mb"] = (
            pre_server.get("gpu_memory_used_mb") if pre_server else None
        )
        batch["gpu_memory_total_mb"] = (
            pre_server.get("gpu_memory_total_mb") if pre_server else None
        )
        batch["gpu_memory_after_server_mb"] = loaded_sample["gpu_memory_used_mb"]
        batch["server_process_memory_after_load_mb"] = loaded_sample[
            "server_process_memory_mb"
        ]

        # One metadata connection distinguishes a healthy server from a dead
        # port before the first multi-worker barrier.
        metadata_client = WebsocketClientPolicy(*endpoint_host_port(endpoint))
        metadata = metadata_client.get_server_metadata()
        close_client(metadata_client)
        batch["server_metadata"] = metadata
        live.write(f"server metadata: {json.dumps(metadata, sort_keys=True)}")

        baseline_infer_ms: float | None = None
        baseline_source_workers: int | None = None
        for workers in args.workers:
            waves = []
            level_gpu_start = len(sampler.snapshot())
            for repeat in range(1, args.repeats + 1):
                sampler.set_context(phase="measured", workers=workers, repeat=repeat)
                live.write(
                    f"level workers={workers} repeat={repeat}/{args.repeats}: "
                    f"{args.requests_per_worker} requests/client"
                )
                wave = run_wave(
                    workers=workers,
                    requests_per_worker=args.requests_per_worker,
                    warmup_per_worker=args.warmup,
                    endpoint=endpoint,
                    payload=payload,
                    repeat=repeat,
                    barrier_timeout_s=args.barrier_timeout_s,
                    request_period_s=(
                        args.request_period_s if args.mode == "periodic" else None
                    ),
                )
                waves.append(wave)
                all_records.extend(wave["records"])
                live.write(
                    f"level workers={workers} repeat={repeat}: "
                    f"requests={wave['requests']} wall={wave['wall_s']:.3f}s "
                    f"throughput={wave['throughput_rps']:.3f} req/s"
                )

            sampler.set_context(phase="between-levels")
            sampler.sample_once("between-levels")
            level_gpu_samples = [
                sample
                for sample in sampler.snapshot()[level_gpu_start:]
                if sample.get("workers") == workers
            ]
            if workers == 1:
                values = [
                    float(record["infer_ms"])
                    for record in all_records
                    if record["workers"] == 1 and record["infer_ms"] is not None
                ]
                baseline_infer_ms = float(np.median(values)) if values else None
                baseline_source_workers = 1 if baseline_infer_ms is not None else None
            elif baseline_infer_ms is None:
                values = [
                    float(record["infer_ms"])
                    for record in all_records
                    if record["workers"] == workers and record["infer_ms"] is not None
                ]
                baseline_infer_ms = float(np.median(values)) if values else None
                baseline_source_workers = (
                    workers if baseline_infer_ms is not None else None
                )
            row = aggregate_level(
                workers=workers,
                waves=waves,
                baseline_infer_ms=baseline_infer_ms,
                gpu_samples=level_gpu_samples,
            )
            level_rows.append(row)
            queue_p95 = row["queue_estimate_ms_p95"]
            queue_text = "-" if queue_p95 is None else f"{queue_p95:.1f}ms"
            live.write(
                f"summary workers={workers}: "
                f"throughput={row['throughput_rps_mean']:.3f} req/s "
                f"client_p95={row['client_latency_ms_p95']:.1f}ms "
                f"infer_p95={row['infer_ms_p95']:.1f}ms "
                f"queue_est_p95={queue_text} "
                f"gpu_util={row['gpu_util_pct_mean']:.1f}% "
                f"gpu_mem={row['gpu_memory_used_mb_mean']:.0f}MiB"
            )

        gpu_samples = sampler.stop()
        sampler = None
    finally:
        if sampler is not None:
            gpu_samples = sampler.stop()
        if server is not None:
            live.write("stopping shared VLA server")
            stop_shared_vla(server)

    seen: set[tuple[Any, ...]] = set()
    unique_gpu_samples: list[dict[str, Any]] = []
    for sample in gpu_samples:
        key = (sample.get("timestamp"), sample.get("phase"), sample.get("workers"))
        if key in seen:
            continue
        seen.add(key)
        unique_gpu_samples.append(sample)

    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    with (run_dir / "gpu_samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in unique_gpu_samples:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")

    throughput_values = [
        row["throughput_rps_mean"]
        for row in level_rows
        if row.get("throughput_rps_mean") is not None
    ]
    max_throughput = max(throughput_values) if throughput_values else None
    throughput_saturation_workers = None
    if max_throughput is not None:
        for row in level_rows:
            value = row.get("throughput_rps_mean")
            if value is not None and value >= 0.95 * max_throughput:
                throughput_saturation_workers = row["workers"]
                break

    summary = {
        "mode": "vla_only_worker_scaling",
        "started_at": batch["started_at"],
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "server_metadata": batch.get("server_metadata"),
        "server_startup_s": batch.get("server_startup_s"),
        "baseline_infer_ms_p50_worker1": (
            level_rows[0]["infer_ms_p50"] if level_rows else None
        ),
        "baseline_source_workers": baseline_source_workers,
        "max_throughput_rps": max_throughput,
        "load_mode": args.mode,
        "request_period_s": (
            args.request_period_s if args.mode == "periodic" else None
        ),
        "throughput_saturation_workers_95pct": throughput_saturation_workers,
        "gpu_memory_pre_server_mb": batch.get("gpu_memory_pre_server_mb"),
        "gpu_memory_after_server_mb": batch.get("gpu_memory_after_server_mb"),
        "gpu_memory_total_mb": batch.get("gpu_memory_total_mb"),
        "server_process_memory_after_load_mb": batch.get(
            "server_process_memory_after_load_mb"
        ),
        "levels": level_rows,
        "notes": [
            "The VLA server serializes inference with an asyncio lock and one executor thread.",
            "Client workers do not add model replicas or persistent GPU memory.",
            "infer_ms is measured from before the server lock and therefore includes FIFO wait.",
            "queue_estimate_ms is infer_ms minus the worker=1 infer_ms median.",
            "periodic mode approximates the observed real workload duty cycle; "
            "saturated mode is an upper-bound stress test.",
        ],
    }
    batch["finished_at"] = summary["finished_at"]
    write_json(run_dir / "batch.json", batch)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", level_rows)
    write_markdown(run_dir / "summary.md", level_rows)
    live.write(f"finished: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
