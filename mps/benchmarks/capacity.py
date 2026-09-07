"""Measure the achievable capacity of the local Apple Silicon GPU through PyTorch MPS.

Reports achieved memory bandwidth, FP16/FP32 matmul throughput, and the wired
memory ceiling so RL capacity decisions can be based on measurements instead of
datasheet guesses. Run from the repository root:

    uv run --project mps python mps/benchmarks/capacity.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import torch

GIB = 1024**3


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()


def measure_bandwidth(size_gib: float = 2.0, repeats: int = 10) -> dict[str, float]:
    """Achieved device bandwidth of a large elementwise copy (read + write)."""
    elements = int(size_gib * GIB / 2)  # float16 -> 2 bytes per element
    source = torch.randn(elements, dtype=torch.float16)
    destination = torch.empty_like(source)
    source = source.to("mps")
    destination = destination.to("mps")
    destination.copy_(source)  # warm up pages
    _sync()

    start = time.perf_counter()
    for _ in range(repeats):
        destination.copy_(source)
    _sync()
    seconds = (time.perf_counter() - start) / repeats
    bytes_moved = destination.numel() * destination.element_size() * 2  # read + write
    return {
        "buffer_gib": size_gib,
        "achieved_gib_s": bytes_moved / seconds / GIB,
        "seconds_per_copy": seconds,
    }


def measure_matmul(m: int, k: int, n: int, dtype: torch.dtype, repeats: int = 20) -> dict[str, float]:
    left = torch.randn(m, k, dtype=dtype)
    right = torch.randn(k, n, dtype=dtype)
    left, right = left.to("mps"), right.to("mps")
    for _ in range(3):  # warmup incl. Metal kernel selection
        left @ right
    _sync()

    start = time.perf_counter()
    for _ in range(repeats):
        left @ right
    _sync()
    seconds = (time.perf_counter() - start) / repeats
    flops = 2 * m * n * k
    return {
        "shape": f"{m}x{k}x{n}",
        "dtype": str(dtype).removeprefix("torch."),
        "tflops": flops / seconds / 1e12,
        "seconds": seconds,
    }


def memory_report(device: torch.device) -> dict[str, float | str]:
    recommended = torch.mps.recommended_max_memory()
    driver_limit = None
    try:
        raw = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, check=False
        )
        if raw.returncode == 0 and raw.stdout.strip():
            driver_limit = int(raw.stdout.strip()) * (1 << 20)
    except OSError:
        pass
    total_ram = 0
    try:
        total_ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    except (OSError, ValueError):
        pass
    return {
        "total_unified_memory_gb": round(total_ram / GIB, 1),
        "recommended_max_working_set_gb": round(recommended / GIB, 1),
        "driver_wired_limit_gb": round(driver_limit / GIB, 1) if driver_limit else "default (~75% of RAM)",
        "allocated_after_benchmarks_gb": round(torch.mps.current_allocated_memory() / GIB, 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON summary")
    args = parser.parse_args(argv)

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; run on Apple Silicon with an MPS-enabled PyTorch")

    device = torch.device("mps")
    torch.manual_seed(0)

    bandwidth = measure_bandwidth()
    matmuls = [
        measure_matmul(4096, 4096, 4096, torch.float32),
        measure_matmul(4096, 4096, 4096, torch.float16),
        measure_matmul(8192, 8192, 8192, torch.float16),
        # llama-style FFN projection at a realistic training shape
        measure_matmul(16384, 4096, 11008, torch.float16),
    ]
    report = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "metal_support": getattr(torch.backends.mps, "is_macos13_or_newer", lambda: None)(),
        "bandwidth": bandwidth,
        "matmul": matmuls,
        "memory": memory_report(device),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    print("CAPACITY_BENCHMARK_OK")


if __name__ == "__main__":
    main()
