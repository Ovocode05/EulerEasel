"""Measure isolated CPU/GPU kernel configurations and rank their timings."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from Testing.algorithm_tuning.benchmark_data import matrix_dimensions
from Testing.algorithm_tuning.settings import (
    CLI_SCRIPT,
    KERNEL_NAMES,
    OMP_KERNELS,
    ROOT,
    ensure_source_path,
)


def worker_run(
    worker_matrix: str,
    worker_kernel: str,
    worker_config: int,
    worker_runs: int,
) -> int:
    """Run one configuration in the subprocess used by the parameter sweep.

    Usage: only the CLI's hidden ``--_worker`` mode calls this function; the
    parent process sets OpenMP and GPU launch settings before starting the worker.
    """
    ensure_source_path()
    import matrix_extractor as me
    import runtime as cpu_runtime
    from Model.context import LazyFrozenContext
    from Model.registry import launch_spmv

    matrix_path = Path(worker_matrix).resolve()
    rows, cols, nnz = me.mat_dim(str(matrix_path))
    context = LazyFrozenContext(str(matrix_path), rows, cols, nnz)
    np.random.seed(7)
    context.x = np.random.default_rng(7).random(cols)
    reference, _ = cpu_runtime.proc_csr(
        context.ensure_cpu_csr(),
        context.x.tolist(),
    )

    if worker_kernel.startswith("GPU_"):
        context.threads = worker_config
        context.blocks = (rows + context.threads - 1) // context.threads

    kernel = getattr(me.Kernel, worker_kernel)
    setup_start = time.perf_counter()
    if worker_kernel == "CPU_CSR_AVX":
        context.ensure_cpu_csr()
    elif worker_kernel.startswith("CPU_ELL"):
        context.ensure_cpu_ell()
    elif worker_kernel == "GPU_CSR":
        context.ensure_gpu_csr()
    elif worker_kernel == "GPU_ELL":
        context.ensure_gpu_ell()
    setup_seconds = time.perf_counter() - setup_start

    launch_spmv(kernel, context)
    latencies = []
    output = None
    for _ in range(worker_runs):
        result = launch_spmv(kernel, context)
        latencies.append(float(result[1] if isinstance(result, (tuple, list)) else result))
        output = result[0] if isinstance(result, (tuple, list)) else None

    if worker_kernel.startswith("GPU_"):
        output = context.d_y.d2h()
    output = np.asarray(output, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    max_abs_error = float(np.max(np.abs(output - reference))) if reference.size else 0.0
    correct = bool(np.allclose(output, reference, rtol=1e-5, atol=1e-8))
    print(
        "TUNING_RESULT="
        + json.dumps(
            {
                "kernel": worker_kernel,
                "config": worker_config,
                "threads": (
                    worker_config
                    if worker_kernel.startswith("GPU_")
                    else int(os.environ.get("OMP_NUM_THREADS", "1"))
                ),
                "blocks": context.blocks if worker_kernel.startswith("GPU_") else None,
                "median_ms": float(median(latencies)),
                "samples_ms": latencies,
                "setup_seconds": setup_seconds,
                "correct": correct,
                "max_abs_error": max_abs_error,
            }
        )
    )
    return 0 if correct else 2


def run_configuration(
    matrix_path: Path,
    kernel: str,
    config: int,
    runs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute and validate one isolated kernel configuration.

    Usage: called by :func:`tune_parameters` for each candidate; subprocess
    failures, timeouts, missing output, or incorrect numerical results raise errors.
    """
    command = [
        sys.executable,
        str(CLI_SCRIPT),
        "--_worker",
        str(matrix_path),
        kernel,
        str(config),
        str(runs),
    ]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(config) if kernel.startswith("CPU_") else "1"

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith("TUNING_RESULT=")
        ),
        None,
    )
    if completed.returncode != 0 or result_line is None:
        raise RuntimeError(
            f"{kernel} config={config} failed (exit={completed.returncode}).\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    result = json.loads(result_line.partition("=")[2])
    if not result["correct"]:
        raise RuntimeError(
            f"{kernel} config={config} produced incorrect output; "
            f"max_abs_error={result['max_abs_error']}"
        )
    return result


def tune_parameters(
    matrix_path: Path,
    kernels: list[str],
    runs: int,
    timeout_seconds: float,
    max_matrix_file_mb: float,
    expected_reuses: int,
) -> None:
    """Sweep supported launch parameters and print best amortized configurations.

    Usage: provide a matrix, kernels, and positive measurement/reuse limits;
    large files are rejected when they exceed the configured size guard.
    """
    matrix_size_mb = matrix_path.stat().st_size / (1024.0 * 1024.0)
    if max_matrix_file_mb > 0 and matrix_size_mb > max_matrix_file_mb:
        raise ValueError(
            f"Refusing parameter sweep: {matrix_path} is {matrix_size_mb:.1f} MiB, "
            f"above the {max_matrix_file_mb:.1f} MiB safety limit."
        )

    rows, cols, nnz = matrix_dimensions(matrix_path)
    logical_cpus = os.cpu_count() or 1
    cpu_configs = sorted(
        {
            1,
            *(2**power for power in range(1, logical_cpus.bit_length()) if 2**power <= logical_cpus),
            logical_cpus,
        }
    )
    gpu_configs = (32, 64, 128, 256, 512)
    results: list[dict[str, Any]] = []

    print(f"\n[Parameter sweep] {matrix_path}")
    print(
        f"  matrix context: rows={rows} cols={cols} nnz={nnz} "
        f"file_size={matrix_size_mb:.2f}MiB logical_cpus={logical_cpus}"
    )
    print(
        "  objective: report median kernel latency and rank correct configurations by "
        f"median + setup/{expected_reuses} expected reuses."
    )

    for kernel in kernels:
        if kernel not in KERNEL_NAMES:
            raise ValueError(f"Unknown kernel {kernel!r}; choose from {', '.join(KERNEL_NAMES)}")
        if kernel.startswith("CPU_") and kernel not in OMP_KERNELS:
            raise ValueError(
                f"{kernel} has no OpenMP thread-count parameter; "
                f"choose one of {', '.join(sorted(OMP_KERNELS))}."
            )
        configs = cpu_configs if kernel in OMP_KERNELS else gpu_configs
        parameter_name = "OMP_NUM_THREADS" if kernel.startswith("CPU_") else "CUDA threads/block"
        print(f"  tuning {kernel}: {parameter_name}={configs}")
        for config in configs:
            result = run_configuration(
                matrix_path,
                kernel,
                config,
                runs,
                timeout_seconds,
            )
            results.append(result)
            print(
                f"    {kernel} config={config:>3} "
                f"median={result['median_ms']:.6f}ms "
                f"setup={result['setup_seconds']:.6f}s "
                f"correct={result['correct']} "
                f"max_abs_error={result['max_abs_error']:.3g} "
                f"blocks={result['blocks']} "
                f"amortized={_amortized_ms(result, expected_reuses):.6f}ms"
            )

    for kernel in kernels:
        candidates = [item for item in results if item["kernel"] == kernel and item["correct"]]
        if not candidates:
            raise RuntimeError(f"No correct configurations completed for {kernel}")
        best = min(candidates, key=lambda item: _amortized_ms(item, expected_reuses))
        parameter_name = "threads/block" if kernel.startswith("GPU_") else "OMP_NUM_THREADS"
        print(
            f"  BEST {kernel}: {parameter_name}={best['config']} "
            f"median={best['median_ms']:.6f}ms "
            f"setup={best['setup_seconds']:.6f}s "
            f"amortized={_amortized_ms(best, expected_reuses):.6f}ms"
        )


def _amortized_ms(result: dict[str, Any], expected_reuses: int) -> float:
    """Combine kernel latency with setup cost amortized over expected reuses.

    Usage: keep candidate ranking and its printed objective consistent across the
    per-configuration and best-configuration summaries.
    """
    return result["median_ms"] + result["setup_seconds"] * 1000 / expected_reuses
