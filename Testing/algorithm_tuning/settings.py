"""Shared paths and kernel metadata for the tuning experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_INCLUDE = ROOT / "Src" / "include"
DEFAULT_LOG = ROOT / "logs" / "benchmark_runtime.log"
DEFAULT_DATASET = ROOT / "dataset"
CLI_SCRIPT = ROOT / "Testing" / "algorithm_tuning_experiment.py"

KERNEL_NAMES = (
    "CPU_CSR",
    "CPU_ELL",
    "CPU_CSR_AVX",
    "CPU_ELL_AVX_x4",
    "CPU_ELL_AVX_x16",
    "GPU_CSR",
    "GPU_ELL",
)
OMP_KERNELS = {"CPU_CSR_AVX", "CPU_ELL_AVX_x4", "CPU_ELL_AVX_x16"}


def ensure_source_path() -> None:
    """Make the source modules and compiled Python bindings importable.

    Usage: call before importing ``Model`` or the extension modules built under
    ``Src/include/native``; each path is added only once.
    """
    for import_path in (SRC_INCLUDE, SRC_INCLUDE / "native"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
