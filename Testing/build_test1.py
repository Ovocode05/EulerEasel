import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC_INCLUDE = ROOT / "Src" / "include"
for import_path in (SRC_INCLUDE, SRC_INCLUDE / "native"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Model.context import LazyFrozenContext
from Model.registry import launch_spmv

DEFAULT_DATASET_ROOTS = [
    ROOT / "dataset",
    ROOT / "Data" / "datasetnaked",
    ROOT / "Data",
]
EXPECTED_HARDWARE_KERNELS = [
    "CPU_CSR",
    "CPU_ELL",
    "CPU_CSR_AVX",
    "CPU_ELL_AVX_x4",
    "CPU_ELL_AVX_x16",
    "GPU_CSR",
    "GPU_ELL",
]


def resolve_matrix_files(dataset_root: str | Path | None = None) -> List[Path]:
    """Return the .mtx matrix files in the configured dataset directory.

    The project currently stores real matrices under the top-level dataset/
    directory, so resolve the actual corpus before falling back to the legacy
    Data/datasetnaked path.
    """
    if dataset_root is not None:
        root = Path(dataset_root)
        if root.exists():
            files = sorted(path for path in root.rglob("*.mtx") if path.is_file())
            return files
        return []

    seen: set[Path] = set()
    for root in DEFAULT_DATASET_ROOTS:
        if root in seen:
            continue
        seen.add(root)
        if root.exists():
            files = sorted(path for path in root.rglob("*.mtx") if path.is_file())
            if files:
                return files

    return []


def _load_runtime_modules():
    try:
        import matrix_extractor as me
        try:
            import runtime as rn  # noqa: F401
        except ModuleNotFoundError:
            rn = None
        try:
            import CUDAruntime as crn  # noqa: F401
        except ModuleNotFoundError:
            crn = None
        return me, rn, crn
    except ModuleNotFoundError:
        return None, None, None


def build_oracle(dataset_root: str | Path | None = None, runs: int = 5):
    """Build a ground-truth oracle for the current dataset.

    This is the Phase 4 benchmark entry point. If the dataset is missing, the
    function returns an empty result instead of crashing the project.
    """
    matrix_files = resolve_matrix_files(dataset_root)
    if not matrix_files:
        return []

    me, _runtime_mod, _cuda_mod = _load_runtime_modules()
    if me is None:
        raise RuntimeError(
            "Runtime modules are not built yet. Run the project build before generating the benchmark oracle."
        )

    str_reg = me.StrategyRegister()
    hrd = me.HardwareContext()
    strategies = str_reg.get_strategies(hrd)
    kernels = [strategy.kernel for strategy in strategies if strategy.is_available]
    kernel_names = [kernel.name for kernel in kernels]
    if not kernel_names:
        raise RuntimeError("No hardware kernels are available on this build; enable the required runtime backend first.")

    oracle_records = []
    for matrix_path in matrix_files:
        [rows, cols, nnz] = me.mat_dim(str(matrix_path))
        frozen_context = LazyFrozenContext(str(matrix_path), rows, cols, nnz)
        kernel_profiles = {}

        for kernel in kernels:
            timings = []
            try:
                for _ in range(runs):
                    result = launch_spmv(kernel, frozen_context)
                    runtime = result[1] if isinstance(result, (tuple, list)) else result
                    timings.append(float(runtime))
            except Exception:
                continue

            if timings:
                kernel_profiles[kernel.name] = float(min(timings))

        if not kernel_profiles:
            continue

        best_kernel_name = min(kernel_profiles, key=kernel_profiles.get)
        oracle_records.append(
            {
                matrix_path.name: {
                    "best_kernel": best_kernel_name,
                    "runtime": kernel_profiles[best_kernel_name],
                    "all_kernel_profiles": kernel_profiles,
                }
            }
        )

    return oracle_records


def main():
    parser = argparse.ArgumentParser(description="EulerEasel benchmark oracle generator")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOTS[0]),
        help="Folder containing matrix files (.mtx) used for the benchmark oracle. Defaults to the workspace dataset folder when present.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Median / best-of-n timing samples per kernel.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    matrix_files = resolve_matrix_files(dataset_root)
    if not matrix_files:
        print(
            f"[Benchmark] No .mtx files found under {dataset_root}. "
            "The dataset is not ready yet; benchmark generation is skipped until the matrix set is available."
        )
        return 0

    try:
        oracle = build_oracle(dataset_root, runs=args.runs)
    except RuntimeError as exc:
        print(f"[Benchmark] {exc}")
        return 1

    if not oracle:
        print("[Benchmark] No valid kernel benchmarks were produced for the current dataset.")
        return 0

    output_path = ROOT / "ground_truth.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(oracle, handle, indent=4)

    print(f"[Benchmark] Oracle generated at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())