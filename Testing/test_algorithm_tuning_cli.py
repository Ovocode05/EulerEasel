"""Tests for argument parsing and output helpers in the tuning CLI."""

import pytest

from Testing.algorithm_tuning.cli import parse_args, print_comparison


def test_parse_args_keeps_default_replay_and_sweep_options() -> None:
    """The no-option CLI uses the documented defaults without enabling tuning."""
    args = parse_args([])

    assert args.alpha == 0.25
    assert args.expected_reuses == 100
    assert args.tune_matrix is None
    assert args.tune_kernels == ["CPU_CSR_AVX", "GPU_CSR"]


def test_parse_args_rejects_nonpositive_reuse_count() -> None:
    """Invalid amortization settings are rejected before any benchmark work."""
    with pytest.raises(SystemExit):
        parse_args(["--expected-reuses", "0"])


def test_print_comparison_reports_lower_regret_policy(capsys) -> None:
    """The summary names whichever replay has lower mean slowdown."""
    result = {
        "top1_accuracy": 0.5,
        "mean_slowdown_percent": 3.0,
        "median_slowdown_percent": 2.0,
        "mean_selected_amortized_ms": 1.25,
    }

    print_comparison(result, {**result, "mean_slowdown_percent": 4.0})

    assert "LinUCB has lower mean regret" in capsys.readouterr().out
