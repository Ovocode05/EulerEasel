"""Command-line argument parsing and orchestration for algorithm tuning."""

import argparse
from pathlib import Path

from Testing.algorithm_tuning.benchmark_data import make_contexts, read_profiles
from Testing.algorithm_tuning.parameter_sweep import tune_parameters, worker_run
from Testing.algorithm_tuning.policy_replay import replay_policy
from Testing.algorithm_tuning.settings import DEFAULT_DATASET, DEFAULT_LOG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options and validate tuning limits.

    Usage: call from :func:`main`, optionally passing an explicit argument list
    when exercising command-line behavior in tests.
    """
    parser = argparse.ArgumentParser(
        description="Compare contextual kernel policies and optionally tune launch parameters."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--alpha", type=float, default=0.25, help="LinUCB exploration coefficient.")
    parser.add_argument("--seed", type=int, default=7, help="LinTS replay random seed.")
    parser.add_argument(
        "--tune-matrix",
        type=Path,
        help="Optional .mtx file for the second-stage CPU/GPU parameter sweep.",
    )
    parser.add_argument(
        "--tune-kernels",
        nargs="+",
        default=["CPU_CSR_AVX", "GPU_CSR"],
        help="Kernels to sweep when --tune-matrix is supplied (default: CPU_CSR_AVX GPU_CSR).",
    )
    parser.add_argument("--tune-runs", type=int, default=7)
    parser.add_argument(
        "--expected-reuses",
        type=int,
        default=100,
        help="Expected SpMV calls per matrix; used to amortize layout/setup time.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-matrix-file-mb",
        type=float,
        default=128.0,
        help="Refuse tuning on files above this size; use 0 to disable the guard.",
    )
    parser.add_argument("--quiet-matrices", action="store_true", help="Print policy summaries only.")
    parser.add_argument(
        "--_worker",
        nargs=4,
        metavar=("MATRIX", "KERNEL", "CONFIG", "RUNS"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if (
        args.alpha <= 0
        or args.tune_runs < 1
        or args.timeout_seconds <= 0
        or args.expected_reuses < 1
        or args.max_matrix_file_mb < 0
    ):
        parser.error(
            "--alpha, --tune-runs, --timeout-seconds, and --expected-reuses must be positive; "
            "--max-matrix-file-mb cannot be negative"
        )
    return args


def print_comparison(ucb_result: dict[str, float], ts_result: dict[str, float]) -> None:
    """Print aggregate policy metrics and identify the lower-regret replay.

    Usage: call after both policies have run to present their comparable summary.
    """
    print("\n=== Policy comparison ===")
    print("policy  | top-1 oracle | mean slowdown | median slowdown | mean amortized ms")
    for policy_name, result in (("LinUCB", ucb_result), ("LinTS", ts_result)):
        print(
            f"{policy_name:<7} | {result['top1_accuracy']:>11.1%} | "
            f"{result['mean_slowdown_percent']:>12.2f}% | "
            f"{result['median_slowdown_percent']:>14.2f}% | "
            f"{result['mean_selected_amortized_ms']:>15.6f}"
        )
    if ucb_result["mean_slowdown_percent"] < ts_result["mean_slowdown_percent"]:
        print("Replay result: LinUCB has lower mean regret on this log; verify on a fresh run.")
    elif ucb_result["mean_slowdown_percent"] > ts_result["mean_slowdown_percent"]:
        print("Replay result: LinTS has lower mean regret on this log; verify on a fresh run.")
    else:
        print("Replay result: tied mean regret on this log; verify on a fresh run.")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark replay and optional parameter sweep.

    Usage: invoke through ``Testing/algorithm_tuning_experiment.py``; an optional
    argument list is accepted for testability and embedding.
    """
    args = parse_args(argv)
    if args._worker:
        return worker_run(
            args._worker[0],
            args._worker[1],
            int(args._worker[2]),
            int(args._worker[3]),
        )

    profiles, setups = read_profiles(args.log)
    matrix_names, contexts, dimensions, layout_features = make_contexts(
        profiles,
        args.dataset_root,
    )
    print("=== EulerEasel policy and parameter experiment ===")
    print(f"benchmark log: {args.log}")
    print(f"correctly measured matrices: {len(matrix_names)}")
    print(
        "model features: log1p(rows), log1p(cols), log1p(nnz), density, "
        "mean row nnz, row CV, ELL fill, log1p(ELL/CSR payload),"
    )
    print("                GPU-available, AVX-available, log1p(logical CPUs)")
    print(
        f"Objective: kernel median + setup amortized over {args.expected_reuses} calls. "
        "This is replay, not held-out validation."
    )

    policy_results = {
        policy_name: replay_policy(
            policy_name,
            matrix_names,
            contexts,
            dimensions,
            layout_features,
            profiles,
            setups,
            args.alpha,
            args.seed,
            args.expected_reuses,
            not args.quiet_matrices,
        )
        for policy_name in ("LinUCB", "LinTS")
    }
    print_comparison(policy_results["LinUCB"], policy_results["LinTS"])

    if args.tune_matrix is not None:
        tune_parameters(
            args.tune_matrix.resolve(),
            args.tune_kernels,
            args.tune_runs,
            args.timeout_seconds,
            args.max_matrix_file_mb,
            args.expected_reuses,
        )
    else:
        print(
            "\nSecond-stage sweep not run. Supply --tune-matrix PATH to measure "
            "OpenMP thread counts and CUDA threads/block."
        )
    return 0
