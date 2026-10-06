"""Replay contextual kernel-selection policies against recorded measurements."""

import math

import numpy as np

from Testing.algorithm_tuning.benchmark_data import (
    LayoutFeatures,
    MatrixDimensions,
    ProfileMap,
    available_kernels,
)
from Testing.algorithm_tuning.settings import KERNEL_NAMES, ensure_source_path


def ucb_arm_scores(model, active_kernels: list[str], context: np.ndarray, alpha: float) -> dict[str, float]:
    """Calculate LinUCB mean-plus-exploration scores for active kernels.

    Usage: call after policy selection when verbose output should explain the
    current LinUCB ranking.
    """
    scores = {}
    for kernel in active_kernels:
        arm = model.models[kernel]
        theta = np.linalg.solve(arm["A"], arm["b"]).reshape(-1)
        uncertainty = math.sqrt(
            max(float(context @ np.linalg.solve(arm["A"], context)), 0.0)
        )
        scores[kernel] = float(context @ theta) + alpha * uncertainty
    return scores


def replay_policy(
    policy_name: str,
    matrix_names: list[str],
    contexts: dict[str, np.ndarray],
    dimensions: dict[str, MatrixDimensions],
    layout_features: dict[str, LayoutFeatures],
    profiles: ProfileMap,
    setups: ProfileMap,
    alpha: float,
    seed: int,
    expected_reuses: int,
    verbose: bool,
) -> dict[str, float]:
    """Replay one contextual bandit and return aggregate oracle/regret metrics.

    Usage: invoke for ``LinUCB`` or ``LinTS`` with contexts and measured timings;
    feedback is limited to the selected kernel's amortized observed latency.
    """
    ensure_source_path()
    from Model.linTS import FeatureNormalizer, LinTS, LinearUCB

    raw_contexts = np.vstack([contexts[name] for name in matrix_names])
    normalizer = FeatureNormalizer().fit(raw_contexts)
    scaled = {
        name: np.concatenate(([1.0], normalizer.transform(contexts[name])[0]))
        for name in matrix_names
    }
    num_features = len(next(iter(scaled.values())))
    model_type = LinearUCB if policy_name == "LinUCB" else LinTS
    policy = model_type(
        kernels=list(KERNEL_NAMES),
        num_features=num_features,
        alpha=alpha,
        ridge=1e-2,
    )
    np.random.seed(seed)

    hit_count = 0
    slowdown_percentages = []
    chosen_latencies = []
    oracle_latencies = []

    print(f"\n[{policy_name}] online replay, seed={seed}")
    for matrix_name in matrix_names:
        available = available_kernels(profiles[matrix_name])
        context = scaled[matrix_name]
        selected, selected_score = policy.choose_kernel(available, context)
        effective_ms = {
            kernel: profiles[matrix_name][kernel]
            + setups[matrix_name][kernel] * 1000.0 / expected_reuses
            for kernel in available
        }
        best_kernel = min(available, key=effective_ms.get)
        chosen_ms = profiles[matrix_name][selected]
        chosen_effective_ms = effective_ms[selected]
        best_effective_ms = effective_ms[best_kernel]
        slowdown = (chosen_effective_ms / best_effective_ms - 1.0) * 100.0

        if selected == best_kernel:
            hit_count += 1
        slowdown_percentages.append(slowdown)
        chosen_latencies.append(chosen_effective_ms)
        oracle_latencies.append(best_effective_ms)

        if verbose:
            _print_matrix_details(
                policy_name,
                matrix_name,
                selected,
                selected_score,
                best_kernel,
                chosen_ms,
                chosen_effective_ms,
                best_effective_ms,
                slowdown,
                context,
                contexts[matrix_name],
                dimensions[matrix_name],
                layout_features[matrix_name],
                profiles[matrix_name],
                setups[matrix_name],
                available,
                policy,
                alpha,
            )

        policy.update(selected, context, reward=1.0 / (1.0 + chosen_effective_ms))

    result = {
        "top1_accuracy": hit_count / len(matrix_names),
        "mean_slowdown_percent": float(np.mean(slowdown_percentages)),
        "median_slowdown_percent": float(np.median(slowdown_percentages)),
        "mean_selected_amortized_ms": float(np.mean(chosen_latencies)),
        "mean_oracle_amortized_ms": float(np.mean(oracle_latencies)),
    }
    print(
        f"  summary: top1={result['top1_accuracy']:.1%}, "
        f"mean slowdown={result['mean_slowdown_percent']:.2f}%, "
        f"median slowdown={result['median_slowdown_percent']:.2f}%"
    )
    return result


def _print_matrix_details(
    policy_name: str,
    matrix_name: str,
    selected: str,
    selected_score: float,
    best_kernel: str,
    chosen_ms: float,
    chosen_effective_ms: float,
    best_effective_ms: float,
    slowdown: float,
    scaled_context: np.ndarray,
    raw_context: np.ndarray,
    dimensions: MatrixDimensions,
    layout_features: LayoutFeatures,
    profile: dict[str, float],
    setup: dict[str, float],
    available: list[str],
    policy,
    alpha: float,
) -> None:
    """Print one matrix's verbose feature, candidate, and decision diagnostics.

    Usage: called internally by :func:`replay_policy` only when verbose output is
    enabled; it keeps presentation logic separate from policy state updates.
    """
    rows, cols, nnz = dimensions
    mean_row_length, row_cv, ell_efficiency, ell_to_csr_ratio = layout_features
    print(
        f"  {matrix_name}: shape={rows}x{cols} nnz={nnz} "
        f"density={raw_context[3]:.6g} mean_row_nnz={mean_row_length:.3f} "
        f"row_cv={row_cv:.3f} ELL_fill={ell_efficiency:.3f} "
        f"ELL/CSR_payload={ell_to_csr_ratio:.3f}"
    )
    print(
        "    measured layout candidates: "
        + ", ".join(
            f"{kernel}={profile[kernel]:.6f}ms + setup {setup[kernel]:.4f}s"
            for kernel in available
        )
    )
    if policy_name == "LinUCB":
        scores = ucb_arm_scores(policy, available, scaled_context, alpha)
        print(
            "    UCB mean+exploration scores: "
            + ", ".join(
                f"{kernel}={scores[kernel]:.5f}"
                for kernel in sorted(scores, key=scores.get, reverse=True)
            )
        )
    print(
        f"    selected={selected} score={selected_score:.5f} "
        f"kernel={chosen_ms:.6f}ms amortized={chosen_effective_ms:.6f}ms "
        f"oracle={best_kernel} amortized={best_effective_ms:.6f}ms "
        f"regret={slowdown:.2f}%"
    )
