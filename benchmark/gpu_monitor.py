#!/usr/bin/env python3
"""Peak GPU memory and utilisation of a server, sampled with nvidia-smi.

The benchmark client is a network client: it never touches the GPU, so the
memory a served model occupies has to be read from outside. Run this next to a
`vllm serve` process for the duration of a benchmark point.

    python benchmark/gpu_monitor.py --duration 300 --out gpu.json

or around a block of code:

    from gpu_monitor import GPUMemoryMonitor
    with GPUMemoryMonitor() as monitor:
        ...
    print(monitor.peak_memory_mib)

nvidia-smi reports the whole device, so a second process on the same GPU is
counted too; a benchmark point is only valid with the GPU otherwise idle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time

QUERY = "memory.used,memory.total,utilization.gpu"


def sample(gpu_index: int = 0) -> dict | None:
    """One nvidia-smi reading, or None if nvidia-smi is absent or fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    fields = [field.strip() for field in out.split(",")]
    if len(fields) != 3:
        return None
    return {
        "t": time.time(),
        "memory_used_mib": float(fields[0]),
        "memory_total_mib": float(fields[1]),
        "utilization_pct": float(fields[2]),
    }


class GPUMemoryMonitor:
    """Background nvidia-smi sampler; peak memory and mean utilisation."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading = sample(self.gpu_index)
            if reading is not None:
                self.samples.append(reading)
            self._stop.wait(self.interval_s)

    def start(self) -> "GPUMemoryMonitor":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 4)
        self._thread = None

    @property
    def peak_memory_mib(self) -> float | None:
        return max((s["memory_used_mib"] for s in self.samples), default=None)

    @property
    def mean_utilization_pct(self) -> float | None:
        if not self.samples:
            return None
        return sum(s["utilization_pct"] for s in self.samples) / len(self.samples)

    def summary(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "interval_s": self.interval_s,
            "samples": len(self.samples),
            "peak_memory_mib": self.peak_memory_mib,
            "memory_total_mib": self.samples[0]["memory_total_mib"] if self.samples else None,
            "mean_utilization_pct": self.mean_utilization_pct,
        }

    def __enter__(self) -> "GPUMemoryMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to sample")
    parser.add_argument("--out", default=None, help="write the summary and samples as JSON")
    args = parser.parse_args()

    monitor = GPUMemoryMonitor(gpu_index=args.gpu_index, interval_s=args.interval)
    with monitor:
        time.sleep(args.duration)
    summary = monitor.summary()
    if not summary["samples"]:
        raise SystemExit("no samples: nvidia-smi is not available on this host")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "samples": monitor.samples}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
