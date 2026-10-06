"""Tests for benchmark parsing and MatrixMarket feature extraction."""

from pathlib import Path

import numpy as np
import pytest

from Testing.algorithm_tuning.benchmark_data import (
    make_contexts,
    matrix_dimensions,
    matrix_layout_features,
    read_profiles,
)


def test_read_profiles_keeps_only_correct_records_and_uses_median(tmp_path: Path) -> None:
    """Successful records become median timings and setup measurements."""
    log_path = tmp_path / "benchmark.log"
    log_path.write_text(
        "matrix=sample.mtx kernel=CPU_CSR status=ok kernel_ms=[2, 4, 3] "
        "setup_seconds=0.25 correct=True\n"
        "matrix=sample.mtx kernel=GPU_CSR status=error kernel_ms=[1] "
        "setup_seconds=0 correct=False\n",
        encoding="utf-8",
    )

    profiles, setups = read_profiles(log_path)

    assert profiles == {"sample.mtx": {"CPU_CSR": 3.0}}
    assert setups == {"sample.mtx": {"CPU_CSR": 0.25}}


def test_read_profiles_rejects_invalid_measurements(tmp_path: Path) -> None:
    """Nonpositive timing samples are rejected with their source line number."""
    log_path = tmp_path / "benchmark.log"
    log_path.write_text(
        "matrix=sample.mtx kernel=CPU_CSR status=ok kernel_ms=[0] "
        "setup_seconds=0 correct=True\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid kernel timing"):
        read_profiles(log_path)


def test_matrix_features_match_small_coordinate_matrix(tmp_path: Path) -> None:
    """Matrix dimensions and layout features use the parsed coordinate rows."""
    matrix_path = tmp_path / "sample.mtx"
    matrix_path.write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        "% tiny test matrix\n"
        "2 3 3\n"
        "1 1 1.0\n"
        "1 3 2.0\n"
        "2 2 3.0\n",
        encoding="utf-8",
    )

    assert matrix_dimensions(matrix_path) == (2, 3, 3)
    assert matrix_layout_features(matrix_path, rows=2) == pytest.approx(
        (1.5, 1 / 3, 0.75, 1.0)
    )


def test_make_contexts_returns_sorted_eleven_feature_vectors(tmp_path: Path) -> None:
    """Context construction sorts matrices and preserves the model feature shape."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("zeta.mtx", "alpha.mtx"):
        (dataset / name).write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "2 2 2\n"
            "1 1 1.0\n"
            "2 2 1.0\n",
            encoding="utf-8",
        )
    profiles = {
        "zeta.mtx": {"CPU_CSR": 2.0},
        "alpha.mtx": {"GPU_CSR": 1.0, "CPU_CSR_AVX": 3.0},
    }

    names, contexts, dimensions, layout = make_contexts(profiles, dataset)

    assert names == ["alpha.mtx", "zeta.mtx"]
    assert contexts["alpha.mtx"].shape == (11,)
    assert np.all(np.isfinite(contexts["alpha.mtx"]))
    assert contexts["alpha.mtx"][8:10].tolist() == [1.0, 1.0]
    assert dimensions["zeta.mtx"] == (2, 2, 2)
    assert layout["zeta.mtx"] == pytest.approx((1.0, 0.0, 1.0, 2 / 3))
