#!/usr/bin/env python3
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_INCLUDE = ROOT / "Src" / "include"
DATASET_ROOT = ROOT / "dataset"
LOG_DIR = ROOT / "logs"


def _runtime_pythonpath() -> str:
    """Build PYTHONPATH entries for source modules and compiled extensions.

    Usage: pass the returned path to subprocesses that import runtime modules.
    Existing PYTHONPATH entries are retained after the repository paths.
    """
    runtime_paths = (SRC_INCLUDE, SRC_INCLUDE / "native")
    existing_paths = os.environ.get("PYTHONPATH")
    paths = [str(path) for path in runtime_paths]
    if existing_paths:
        paths.append(existing_paths)
    return os.pathsep.join(paths)


def ensure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
    )
    return logging.getLogger("eulereasel")


def discover_valid_matrices(dataset_root: Path | str) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    files = sorted(path for path in root.rglob("*.mtx") if path.is_file())
    return [path for path in files if "label" not in path.name.lower()]


def _balanced_json_fragment(text: str, start: int):
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _extract_first_json(payload: str):
    text = payload.strip()
    for start in range(len(text)):
        if text[start] not in "[{":
            continue
        candidate = _balanced_json_fragment(text, start)
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON payload found in output: {payload[:300]!r}")


def discover_available_kernels() -> list[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _runtime_pythonpath()
    code = """
import json
import matrix_extractor as me
hrd = me.HardwareContext()
str_reg = me.StrategyRegister()
strategies = str_reg.get_strategies(hrd)
print(json.dumps([s.kernel.name for s in strategies if s.is_available]))
"""
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "kernel discovery failed")
    return _extract_first_json(proc.stdout)


def run_single_matrix_kernel(
    matrix_path: Path,
    kernel_name: str,
    runs: int,
    timeout_seconds: float,
) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _runtime_pythonpath()
    code = f"""
import json, time, traceback
import matrix_extractor as me
import runtime as cpu_runtime
from Model.context import LazyFrozenContext
from Model.registry import launch_spmv

matrix_path = {str(matrix_path)!r}
kernel_name = {kernel_name!r}
try:
    rows, cols, nnz = me.mat_dim(matrix_path)
    ctx = LazyFrozenContext(matrix_path, rows, cols, nnz)

    # Build a reference and all required format/device state before timing.
    reference, _ = cpu_runtime.proc_csr(ctx.ensure_cpu_csr(), ctx.x.tolist())
    setup_start = time.perf_counter()
    if kernel_name in ('CPU_CSR', 'CPU_CSR_AVX'):
        ctx.ensure_cpu_csr()
    elif kernel_name.startswith('CPU_ELL'):
        ctx.ensure_cpu_ell()
    elif kernel_name == 'GPU_CSR':
        ctx.ensure_gpu_csr()
    elif kernel_name == 'GPU_ELL':
        ctx.ensure_gpu_ell()
    setup_seconds = time.perf_counter() - setup_start

    kernel = getattr(me.Kernel, kernel_name)
    launch_spmv(kernel, ctx)

    def kernel_ms(result):
        if isinstance(result, (tuple, list)):
            return float(result[1])
        return float(result)

    latencies_ms = []
    wall_latencies_ms = []
    output = None
    for _ in range({runs}):
        start = time.perf_counter()
        result = launch_spmv(kernel, ctx)
        wall_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        latencies_ms.append(kernel_ms(result))
        output = result[0] if isinstance(result, (tuple, list)) else None

    import numpy as np
    if kernel_name.startswith('GPU_'):
        output = ctx.d_y.d2h()
    correct = bool(np.allclose(output, reference, rtol=1e-5, atol=1e-8))
    print(json.dumps({{
        'kernel': kernel_name,
        'status': 'ok' if correct else 'incorrect',
        'kernel_latencies_ms': latencies_ms,
        'wall_latencies_ms': wall_latencies_ms,
        'setup_seconds': setup_seconds,
        'correct': correct,
        'max_abs_error': float(np.max(np.abs(np.asarray(output) - np.asarray(reference)))) if len(reference) else 0.0,
    }}))
except BaseException as exc:
    print(json.dumps({{
        'kernel': kernel_name,
        'status': 'error',
        'error_type': type(exc).__name__,
        'stderr': str(exc),
        'traceback': traceback.format_exc(),
    }}))
    raise
"""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_output = exc.stderr or exc.stdout or ""
        if isinstance(timeout_output, bytes):
            timeout_output = timeout_output.decode(errors="replace")
        return {
            "kernel": kernel_name,
            "status": "timeout",
            "issue": f"exceeded {timeout_seconds:g} seconds",
            "stderr": timeout_output.strip(),
            "wall_seconds": time.perf_counter() - start,
        }

    wall_seconds = time.perf_counter() - start
    if proc.returncode != 0:
        child_error = {}
        try:
            parsed = _extract_first_json(proc.stdout)
            if isinstance(parsed, dict) and parsed.get("status") == "error":
                child_error = parsed
        except ValueError:
            pass
        signal_name = None
        if proc.returncode < 0:
            try:
                signal_name = signal.Signals(-proc.returncode).name
            except ValueError:
                signal_name = f"signal {-proc.returncode}"
        return {
            "kernel": kernel_name,
            "status": "error",
            "issue": (
                f"{child_error.get('error_type')}: {child_error.get('stderr')}"
                if child_error
                else f"native process terminated by {signal_name}" if signal_name else f"process exited {proc.returncode}"
            ),
            "stderr": child_error.get("traceback", "") or (proc.stderr or proc.stdout).strip() or "process produced no diagnostic output",
            "returncode": proc.returncode,
            "wall_seconds": wall_seconds,
        }

    payload = proc.stdout.strip()
    if not payload:
        return {"kernel": kernel_name, "status": "error", "stderr": "empty output"}
    try:
        data = _extract_first_json(payload)
    except ValueError:
        return {"kernel": kernel_name, "status": "error", "stderr": payload}

    data["wall_seconds"] = wall_seconds
    return data


def summarize(results: list[dict]) -> dict:
    rows = []
    oracle_by_matrix: dict[str, dict[str, float]] = {}
    for item in results:
        if item.get("status") in ("ok", "incorrect"):
            latencies = item.get("kernel_latencies_ms", [])
            if not latencies:
                rows.append({
                    "matrix": item["matrix"],
                    "kernel": item["kernel"],
                    "status": "error",
                    "issue": "no kernel latency samples returned",
                })
                continue
            rows.append({
                "matrix": item["matrix"],
                "kernel": item["kernel"],
                "status": item.get("status", "ok"),
                "best_ms": min(latencies),
                "median_ms": sorted(latencies)[len(latencies) // 2],
                "mean_ms": sum(latencies) / len(latencies),
                "setup_seconds": item.get("setup_seconds"),
                "wall_seconds": item.get("wall_seconds"),
                "correct": item.get("correct"),
                "max_abs_error": item.get("max_abs_error"),
            })
            if item.get("status") == "ok" and item.get("correct"):
                oracle_by_matrix.setdefault(item["matrix"], {})[item["kernel"]] = sorted(latencies)[len(latencies) // 2]
        else:
            rows.append({
                "matrix": item["matrix"],
                "kernel": item["kernel"],
                "status": item.get("status", "error"),
                "issue": item.get("issue", item.get("error_type", "benchmark failed")),
                "stderr": item.get("stderr", ""),
                "returncode": item.get("returncode"),
            })

    oracle = {
        matrix: {
            "best_kernel": min(profiles, key=profiles.get),
            "runtime_ms": min(profiles.values()),
            "all_kernel_profiles_ms": profiles,
        }
        for matrix, profiles in oracle_by_matrix.items()
    }
    return {"entries": rows, "oracle": oracle}


def main():
    parser = argparse.ArgumentParser(description="EulerEasel kernel benchmark evaluator")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-matrix-file-mb",
        type=float,
        default=128.0,
        help="Skip larger input files to avoid exhausting host memory; use 0 to disable the size guard.",
    )
    parser.add_argument("--log-path", type=str, default=str(LOG_DIR / "benchmark_runtime.log"))
    parser.add_argument("--output", type=str, default=str(ROOT / "benchmark_results.json"))
    args = parser.parse_args()

    log_path = Path(args.log_path)
    logger = ensure_logging(log_path)
    logger.info("=== benchmark start ===")
    logger.info(
        "dataset_root=%s runs=%s timeout_seconds=%s max_matrix_file_mb=%s",
        args.dataset_root,
        args.runs,
        args.timeout_seconds,
        args.max_matrix_file_mb,
    )

    matrices = discover_valid_matrices(args.dataset_root)
    if not matrices:
        logger.warning("No valid matrices found for benchmark")
        print("No valid matrices found for benchmark")
        return 1

    try:
        kernels = discover_available_kernels()
    except Exception as exc:
        logger.exception("kernel discovery failed: %s", exc)
        print(f"Kernel discovery failed: {exc}")
        return 2

    logger.info("Available kernels=%s", kernels)
    print(f"Valid matrices: {len(matrices)}")
    print(f"Available kernels: {kernels}")

    results = []
    summary_path = Path(args.output)
    for matrix_path in matrices:
        logger.info("matrix=%s start", matrix_path.name)
        file_size_mb = matrix_path.stat().st_size / (1024.0 * 1024.0)
        for kernel_name in kernels:
            if args.max_matrix_file_mb > 0 and file_size_mb > args.max_matrix_file_mb:
                result = {
                    "matrix": matrix_path.name,
                    "kernel": kernel_name,
                    "status": "skipped_resource_limit",
                    "issue": f"file size {file_size_mb:.1f} MiB exceeds configured limit {args.max_matrix_file_mb:.1f} MiB",
                }
                results.append(result)
                logger.warning(
                    "matrix=%s kernel=%s status=%s issue=%s",
                    matrix_path.name,
                    kernel_name,
                    result["status"],
                    result["issue"],
                )
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summarize(results), indent=2), encoding="utf-8")
                continue

            logger.info("matrix=%s kernel=%s start", matrix_path.name, kernel_name)
            try:
                result = run_single_matrix_kernel(
                    matrix_path,
                    kernel_name,
                    runs=args.runs,
                    timeout_seconds=args.timeout_seconds,
                )
                result["matrix"] = matrix_path.name
                result["file_size_mb"] = file_size_mb
                results.append(result)
                if result.get("status") in ("ok", "incorrect"):
                    latencies = result.get("kernel_latencies_ms", [])
                    logger.info(
                        "matrix=%s kernel=%s status=%s kernel_ms=%s wall_latencies_ms=%s setup_seconds=%.6f wall_seconds=%.6f correct=%s max_abs_error=%s",
                        matrix_path.name,
                        kernel_name,
                        result.get("status"),
                        [round(x, 6) for x in latencies],
                        [round(x, 6) for x in result.get("wall_latencies_ms", [])],
                        result.get("setup_seconds", 0.0),
                        result.get("wall_seconds", 0.0),
                        result.get("correct"),
                        result.get("max_abs_error"),
                    )
                else:
                    logger.warning(
                        "matrix=%s kernel=%s status=%s issue=%s returncode=%s stderr=%s",
                        matrix_path.name,
                        kernel_name,
                        result.get("status"),
                        result.get("issue", ""),
                        result.get("returncode", ""),
                        result.get("stderr", ""),
                    )
            except Exception as exc:
                issue = {"matrix": matrix_path.name, "kernel": kernel_name, "status": "error", "stderr": f"{type(exc).__name__}: {exc}"}
                results.append(issue)
                logger.exception("matrix=%s kernel=%s benchmark crashed", matrix_path.name, kernel_name)
            summary = summarize(results)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("matrix=%s complete", matrix_path.name)

    summary = summarize(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=== benchmark end ===")
    for item in summary["entries"]:
        if item.get("status") not in ("ok", "incorrect"):
            print(f"{item['status'].upper()} {item['matrix']} {item['kernel']} => {item.get('issue', item.get('stderr',''))[:200]}")
        else:
            print(f"{item['matrix']} | {item['kernel']} | best={item['best_ms']:.4f}ms median={item['median_ms']:.4f}ms mean={item['mean_ms']:.4f}ms | status={item['status']} correct={item['correct']}")

    logger.info("oracle matrices=%s", len(summary["oracle"]))
    for matrix, oracle in summary["oracle"].items():
        logger.info("oracle matrix=%s best_kernel=%s runtime_ms=%.6f", matrix, oracle["best_kernel"], oracle["runtime_ms"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
