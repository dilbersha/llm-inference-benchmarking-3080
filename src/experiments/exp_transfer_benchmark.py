"""
Experiment 4A: CPU↔GPU Memory Transfer Benchmark

Question: What's the real cost of offloading KV cache to CPU RAM?
How much latency does PCIe transfer add at various cache sizes?
"""

from __future__ import annotations

import time

import torch

from src.experiments.runner import ExperimentRunner


def benchmark_transfer(
    size_mb: float,
    direction: str,
    device: str = "cuda",
    use_pinned: bool = False,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> dict:
    """Benchmark a single GPU↔CPU transfer."""
    num_elements = int(size_mb * 1024 * 1024 / 4)  # float32

    if direction == "gpu_to_cpu":
        if device == "cuda" and torch.cuda.is_available():
            src = torch.randn(num_elements, device="cuda")
            if use_pinned:
                dst = torch.empty(num_elements, pin_memory=True)
            else:
                dst = torch.empty(num_elements)
        else:
            # CPU-only simulation
            src = torch.randn(num_elements)
            dst = torch.empty(num_elements)
    else:  # cpu_to_gpu
        if use_pinned:
            src = torch.empty(num_elements, pin_memory=True)
            torch.randn(num_elements, out=src)
        else:
            src = torch.randn(num_elements)

        if device == "cuda" and torch.cuda.is_available():
            dst = torch.empty(num_elements, device="cuda")
        else:
            dst = torch.empty(num_elements)

    # Warmup
    for _ in range(num_warmup):
        if device == "cuda" and torch.cuda.is_available():
            if direction == "gpu_to_cpu":
                dst.copy_(src)
                torch.cuda.synchronize()
            else:
                dst.copy_(src)
                torch.cuda.synchronize()
        else:
            dst.copy_(src)

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start = time.perf_counter()
            dst.copy_(src)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        else:
            start = time.perf_counter()
            dst.copy_(src)
            elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    bandwidth_gbps = [size_mb / 1024 / t for t in times]  # GB/s

    return {
        "transfer_time_ms_mean": sum(times_ms) / len(times_ms),
        "transfer_time_ms_min": min(times_ms),
        "transfer_time_ms_max": max(times_ms),
        "transfer_time_ms_p50": sorted(times_ms)[len(times_ms) // 2],
        "bandwidth_gbps_mean": sum(bandwidth_gbps) / len(bandwidth_gbps),
        "bandwidth_gbps_max": max(bandwidth_gbps),
    }


def run_transfer_benchmark():
    """Run CPU↔GPU transfer benchmark."""
    runner = ExperimentRunner(
        name="memory_transfer_benchmark",
        description=(
            "Benchmarks PCIe transfer latency for KV cache offloading. "
            "Tests various cache sizes, directions, and pinned vs. paged memory."
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sizes_mb = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    directions = ["gpu_to_cpu", "cpu_to_gpu"]
    pinned_options = [False, True] if device == "cuda" else [False]

    for size_mb in sizes_mb:
        for direction in directions:
            for use_pinned in pinned_options:
                config = {
                    "size_mb": size_mb,
                    "direction": direction,
                    "use_pinned_memory": use_pinned,
                    "device": device,
                }

                with runner.trial(config) as trial:
                    result = benchmark_transfer(
                        size_mb=size_mb,
                        direction=direction,
                        device=device,
                        use_pinned=use_pinned,
                    )
                    for k, v in result.items():
                        trial.record(k, v)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_transfer_benchmark()
