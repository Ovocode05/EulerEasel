import sys
from pathlib import Path

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Src" / "include"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model.context import build_runtime_context_vector
from Model.linTS import FeatureNormalizer, HierarchicalKernelSelector, LinearUCB

BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_module",
    ROOT / "Testing" / "build_test1.py",
)
benchmark_module = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark_module)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "kernel_benchmark_module",
    ROOT / "Testing" / "run_kernel_benchmark.py",
)
runner_module = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner_module)


def test_feature_normalizer_stabilizes_inputs():
    normalizer = FeatureNormalizer()
    data = np.array([[1.0, 100.0], [3.0, 120.0], [5.0, 140.0]], dtype=np.float64)
    transformed = normalizer.fit_transform(data)

    assert transformed.shape == data.shape
    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)


def test_linear_ucb_prefers_high_reward_arm():
    kernels = ["cpu_csr", "gpu_csr"]
    bandit = LinearUCB(kernels=kernels, num_features=2, alpha=1.0)

    x = np.array([1.0, 0.0], dtype=np.float64)
    bandit.update("cpu_csr", x, reward=1.0)
    bandit.update("cpu_csr", x, reward=1.2)
    bandit.update("gpu_csr", x, reward=0.2)
    bandit.update("gpu_csr", x, reward=0.1)

    chosen, _ = bandit.choose_kernel(kernels, x)
    assert chosen == "cpu_csr"


def test_runtime_context_vector_includes_hardware_state():
    matrix_features = np.array([1000.0, 500.0, 9000.0, 8.0, 10.0], dtype=np.float64)
    hardware = {
        "has_gpu": 1.0,
        "has_avx": 1.0,
        "logical_threads": 16.0,
        "gpu_memory_gb": 8.0,
        "cpu_memory_gb": 32.0,
    }

    context = build_runtime_context_vector(matrix_features, hardware)

    assert context.shape[0] == matrix_features.shape[0] + 5
    assert np.all(np.isfinite(context))
    assert context[-1] == 32.0
    assert context[-2] == 8.0


def test_hierarchical_selector_prefers_backend_then_kernel():
    selector = HierarchicalKernelSelector(
        backend_names=["cpu", "gpu"],
        layout_kernels_by_backend={
            "cpu": ["cpu_csr", "cpu_ell"],
            "gpu": ["gpu_csr", "gpu_ell"],
        },
        num_features=2,
        alpha=1.0,
    )

    x = np.array([1.0, 0.0], dtype=np.float64)
    selector.update(x, "cpu", "cpu_csr", reward=2.0)
    selector.update(x, "cpu", "cpu_csr", reward=1.5)
    selector.update(x, "gpu", "gpu_csr", reward=0.3)
    selector.update(x, "gpu", "gpu_ell", reward=0.1)

    backend, kernel, _ = selector.select(x)
    assert backend == "cpu"
    assert kernel == "cpu_csr"


def test_benchmark_harness_uses_full_hardware_kernel_family():
    expected = [
        "CPU_CSR",
        "CPU_ELL",
        "CPU_CSR_AVX",
        "CPU_ELL_AVX_x4",
        "CPU_ELL_AVX_x16",
        "GPU_CSR",
        "GPU_ELL",
    ]
    assert benchmark_module.EXPECTED_HARDWARE_KERNELS == expected


def test_benchmark_harness_handles_missing_dataset_without_crashing(tmp_path):
    missing_dir = tmp_path / "missing_dataset"
    matrices = benchmark_module.resolve_matrix_files(missing_dir)
    assert matrices == []


def test_benchmark_summary_uses_kernel_latency_not_setup_time():
    summary = runner_module.summarize(
        [
            {
                "matrix": "sample.mtx",
                "kernel": "CPU_CSR",
                "status": "ok",
                "kernel_latencies_ms": [2.0, 4.0, 3.0],
                "setup_seconds": 9.0,
            }
        ]
    )

    entry = summary["entries"][0]
    assert entry["median_ms"] == 3.0
    assert entry["mean_ms"] == 3.0
    assert entry["setup_seconds"] == 9.0


def test_benchmark_oracle_uses_only_correct_kernels():
    summary = runner_module.summarize(
        [
            {
                "matrix": "sample.mtx",
                "kernel": "CPU_CSR",
                "status": "ok",
                "correct": True,
                "kernel_latencies_ms": [2.0, 3.0, 2.5],
            },
            {
                "matrix": "sample.mtx",
                "kernel": "GPU_CSR",
                "status": "incorrect",
                "correct": False,
                "kernel_latencies_ms": [0.1, 0.2, 0.1],
            },
        ]
    )

    assert summary["oracle"]["sample.mtx"]["best_kernel"] == "CPU_CSR"
    assert summary["oracle"]["sample.mtx"]["runtime_ms"] == 2.5
