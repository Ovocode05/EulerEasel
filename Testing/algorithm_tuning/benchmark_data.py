"""Parse benchmark results and build matrix contexts for policy replay."""

import ast
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import TypeAlias

import numpy as np

ProfileMap: TypeAlias = dict[str, dict[str, float]]
MatrixDimensions: TypeAlias = tuple[int, int, int]
LayoutFeatures: TypeAlias = tuple[float, float, float, float]

LOG_RECORD = re.compile(
    r"matrix=(?P<matrix>\S+) kernel=(?P<kernel>\S+) "
    r"status=(?P<status>\S+) kernel_ms=(?P<times>\[[^\]]*\]).*?"
    r"setup_seconds=(?P<setup>[0-9.eE+-]+).*?"
    r"correct=(?P<correct>True|False)"
)


def read_profiles(log_path: Path) -> tuple[ProfileMap, ProfileMap]:
    """Return median timings and setup seconds for successful log records.

    Usage: pass the benchmark log path before constructing contextual features;
    invalid timing samples and logs without valid measurements raise an error.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Benchmark log does not exist: {log_path}")

    profiles: ProfileMap = defaultdict(dict)
    setups: ProfileMap = defaultdict(dict)
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        match = LOG_RECORD.search(line)
        if match is None or match["status"] != "ok" or match["correct"] != "True":
            continue
        samples = ast.literal_eval(match["times"])
        if not samples or any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in samples
        ):
            raise ValueError(f"Invalid kernel timing on {log_path}:{line_number}")
        profiles[match["matrix"]][match["kernel"]] = float(median(samples))
        setup_seconds = float(match["setup"])
        if not math.isfinite(setup_seconds) or setup_seconds < 0:
            raise ValueError(f"Invalid setup time on {log_path}:{line_number}")
        setups[match["matrix"]][match["kernel"]] = setup_seconds

    if not profiles:
        raise ValueError(f"No successful/correct kernel measurements found in {log_path}")
    return dict(profiles), dict(setups)


def matrix_dimensions(matrix_path: Path) -> MatrixDimensions:
    """Read MatrixMarket dimensions without loading matrix entries.

    Usage: call when only row, column, and nonzero counts are needed, including
    before a parameter sweep that may inspect a large matrix.
    """
    with matrix_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            fields = stripped.split()
            if len(fields) < 3:
                break
            try:
                rows, cols, nnz = map(int, fields[:3])
            except ValueError:
                break
            if rows <= 0 or cols <= 0 or nnz < 0:
                break
            return rows, cols, nnz
    raise ValueError(f"Could not parse MatrixMarket dimensions from {matrix_path}")


def matrix_layout_features(matrix_path: Path, rows: int) -> LayoutFeatures:
    """Estimate row skew, ELL fill, and raw value/index storage from entries.

    Usage: provide the MatrixMarket path and its row count to derive the four
    layout features printed by verbose policy replay.
    """
    row_counts: dict[int, int] = defaultdict(int)
    dimension_line_seen = False
    entry_count = 0

    with matrix_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if not dimension_line_seen:
                dimension_line_seen = True
                continue
            fields = stripped.split()
            if len(fields) < 3:
                continue
            try:
                row = int(fields[0]) - 1
                int(fields[1])
                float(fields[2])
            except ValueError:
                continue
            if not 0 <= row < rows:
                raise ValueError(
                    f"Out-of-range row index in {matrix_path}:{line_number}: {row + 1}"
                )
            row_counts[row] += 1
            entry_count += 1

    if entry_count == 0:
        raise ValueError(f"No parseable coordinate entries found in {matrix_path}")

    mean_row_length = entry_count / rows
    sum_squares = sum(count * count for count in row_counts.values())
    variance = max(sum_squares / rows - mean_row_length * mean_row_length, 0.0)
    row_cv = math.sqrt(variance) / mean_row_length if mean_row_length else 0.0
    max_row_length = max(row_counts.values())
    ell_efficiency = entry_count / (rows * max_row_length)
    csr_payload_bytes = entry_count * 12 + (rows + 1) * 4
    ell_payload_bytes = rows * max_row_length * 12
    ell_to_csr_payload_ratio = ell_payload_bytes / csr_payload_bytes
    return mean_row_length, row_cv, ell_efficiency, ell_to_csr_payload_ratio


def resolve_matrix_path(dataset_root: Path, matrix_name: str) -> Path:
    """Find one unambiguous dataset matrix by its logged file name.

    Usage: use for resolving matrix names in benchmark records to source files;
    missing or duplicate matches are reported instead of silently choosing one.
    """
    matches = sorted(path for path in dataset_root.rglob(matrix_name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No matrix named {matrix_name} under {dataset_root}")
    if len(matches) > 1:
        raise ValueError(
            f"Matrix name {matrix_name!r} is ambiguous under {dataset_root}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def make_contexts(
    profiles: ProfileMap,
    dataset_root: Path,
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, MatrixDimensions],
    dict[str, LayoutFeatures],
]:
    """Build fixed-scale matrix and hardware features for policy replay.

    Usage: provide per-matrix kernel profiles and the dataset root; returns the
    sorted matrix names, model contexts, dimensions, and layout display features.
    """
    matrix_names = sorted(profiles)
    contexts: dict[str, np.ndarray] = {}
    dimensions: dict[str, MatrixDimensions] = {}
    layout_features: dict[str, LayoutFeatures] = {}
    logical_cpus = os.cpu_count() or 1

    for matrix_name in matrix_names:
        matrix_path = resolve_matrix_path(dataset_root, matrix_name)
        rows, cols, nnz = matrix_dimensions(matrix_path)
        dimensions[matrix_name] = (rows, cols, nnz)
        mean_row_length, row_cv, ell_efficiency, ell_to_csr_ratio = matrix_layout_features(
            matrix_path,
            rows,
        )
        layout_features[matrix_name] = (
            mean_row_length,
            row_cv,
            ell_efficiency,
            ell_to_csr_ratio,
        )
        has_gpu = float(any(name.startswith("GPU_") for name in profiles[matrix_name]))
        has_avx = float(any("AVX" in name for name in profiles[matrix_name]))
        density = nnz / (rows * cols)

        # These values correspond to the eleven features described in the CLI output.
        contexts[matrix_name] = np.asarray(
            [
                math.log1p(rows),
                math.log1p(cols),
                math.log1p(nnz),
                density,
                math.log1p(mean_row_length),
                row_cv,
                ell_efficiency,
                math.log1p(ell_to_csr_ratio),
                has_gpu,
                has_avx,
                math.log1p(logical_cpus),
            ],
            dtype=np.float64,
        )
    return matrix_names, contexts, dimensions, layout_features


def available_kernels(profile: dict[str, float]) -> list[str]:
    """Return profiled kernels in their original log insertion order.

    Usage: use this helper before policy selection or display. Preserving insertion
    order retains the selection/tie-breaking behavior of the original replay.
    """
    return list(profile)
