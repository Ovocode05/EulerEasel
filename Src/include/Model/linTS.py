import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, "/home/fakeheadset/Projects/EulerEasel/Src/include")


class FeatureNormalizer:
    """Standardize a feature matrix while keeping the implementation simple and stable."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "FeatureNormalizer":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        self.mean_ = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.scale_ = std
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Normalizer has not been fitted yet.")

        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return (matrix - self.mean_) / self.scale_

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).transform(matrix)


class LinearUCB:
    """Linear UCB contextual bandit for kernel selection.

    This is a cleaner replacement for the earlier single flat LinTS baseline.
    It keeps the model per arm, regularizes the covariance matrix, and scores each
    kernel using the optimistic upper confidence bound.
    """

    def __init__(self, kernels: Sequence[object], num_features: int, alpha: float = 1.0, ridge: float = 1e-3):
        self.kernels = list(kernels)
        self.num_features = int(num_features)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.models: Dict[object, Dict[str, np.ndarray]] = {}

        identity = np.eye(self.num_features, dtype=np.float64)
        for kernel in self.kernels:
            self.models[kernel] = {
                "A": identity.copy() * (1.0 + self.ridge),
                "b": np.zeros((self.num_features, 1), dtype=np.float64),
            }

    @staticmethod
    def _prepare_feature_vector(x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        vector = np.asarray(x, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            raise ValueError("Feature vector cannot be empty.")
        return vector

    def _validate_kernel(self, kernel: object) -> None:
        if kernel not in self.models:
            raise KeyError(f"Kernel {kernel!r} is not registered in this bandit.")

    def _theta(self, kernel: object) -> np.ndarray:
        self._validate_kernel(kernel)
        model = self.models[kernel]
        A = model["A"]
        b = model["b"]
        return np.linalg.solve(A, b).reshape(-1)

    def choose_kernel(self, active_kernels: Sequence[object], x: Union[np.ndarray, Sequence[float]]) -> Tuple[object, float]:
        feature_vector = self._prepare_feature_vector(x)
        if feature_vector.size != self.num_features:
            raise ValueError(
                f"Expected feature length {self.num_features}, got {feature_vector.size}."
            )

        score_map: Dict[object, float] = {}
        for kernel in active_kernels:
            self._validate_kernel(kernel)
            model = self.models[kernel]
            A = model["A"]
            b = model["b"]
            theta = np.linalg.solve(A, b).reshape(-1)
            inv_term = np.linalg.inv(A)
            mean = float(feature_vector @ theta)
            uncertainty = float(np.sqrt(max(feature_vector @ inv_term @ feature_vector, 0.0)))
            score_map[kernel] = mean + self.alpha * uncertainty

        best_kernel = max(score_map, key=score_map.get)
        return best_kernel, float(score_map[best_kernel])

    def update(self, kernel: object, x: Union[np.ndarray, Sequence[float]], reward: float) -> None:
        self._validate_kernel(kernel)
        feature_vector = self._prepare_feature_vector(x)
        if feature_vector.size != self.num_features:
            raise ValueError(
                f"Expected feature length {self.num_features}, got {feature_vector.size}."
            )

        model = self.models[kernel]
        column = feature_vector.reshape(-1, 1)
        model["A"] += column @ column.T + self.ridge * np.eye(self.num_features, dtype=np.float64)
        model["b"] += column * float(reward)

    def get_best_kernel(self, kernels: Sequence[object], x: Union[np.ndarray, Sequence[float]]) -> List[Tuple[object, float]]:
        feature_vector = self._prepare_feature_vector(x)
        expected_rewards: Dict[object, float] = {}

        for kernel in kernels:
            self._validate_kernel(kernel)
            theta = self._theta(kernel)
            expected_rewards[kernel] = float(feature_vector @ theta)

        return sorted(expected_rewards.items(), key=lambda item: item[1], reverse=True)

    def save(self, filepath: str) -> None:
        save_dict = {}
        for kernel, matrices in self.models.items():
            kernel_name = getattr(kernel, "name", str(kernel))
            save_dict[f"model__{kernel_name}__A"] = matrices["A"]
            save_dict[f"model__{kernel_name}__b"] = matrices["b"]

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: str, enum_class) -> None:
        self.models = {}
        with np.load(filepath) as data:
            for key in data.files:
                if not key.startswith("model__"):
                    continue

                parts = key.split("__")
                if len(parts) != 3:
                    continue

                kernel_name = parts[1]
                matrix_name = parts[2]
                kernel_enum = getattr(enum_class, kernel_name, None)
                if kernel_enum is None:
                    continue

                if kernel_enum not in self.models:
                    self.models[kernel_enum] = {}
                self.models[kernel_enum][matrix_name] = data[key]


class LinTS(LinearUCB):
    """Backward-compatible Thompson-sampling implementation.

    Phase 1 keeps the older API, but the actual policy is now implemented in a cleaner,
    reusable base class. This avoids duplicate logic between UCB and TS variants.
    """

    def __init__(self, kernels: Sequence[object], num_features: int, alpha: float = 1.0, ridge: float = 1e-3):
        super().__init__(kernels=kernels, num_features=num_features, alpha=alpha, ridge=ridge)

    def choose_kernel(self, active_kernels: Sequence[object], x: Union[np.ndarray, Sequence[float]]) -> Tuple[object, float]:
        feature_vector = self._prepare_feature_vector(x)
        if feature_vector.size != self.num_features:
            raise ValueError(
                f"Expected feature length {self.num_features}, got {feature_vector.size}."
            )

        sampled_scores: Dict[object, float] = {}
        for kernel in active_kernels:
            self._validate_kernel(kernel)
            model = self.models[kernel]
            A = model["A"]
            b = model["b"]

            theta = np.linalg.solve(A, b).reshape(-1)
            L = np.linalg.cholesky(A)
            z = np.random.standard_normal(len(theta))
            noise = np.linalg.solve(L.T, z)
            sampled_theta = theta + noise
            sampled_scores[kernel] = float(feature_vector @ sampled_theta)

        best_kernel = max(sampled_scores, key=sampled_scores.get)
        return best_kernel, float(sampled_scores[best_kernel])


class HierarchicalKernelSelector:
    """Select backend and format in two stages.

    This mirrors the runtime architecture: first decide the hardware family and then
    choose the most suitable kernel inside that family. It keeps the policy simple,
    explainable, and easy to validate before we move to a heavier model.
    """

    def __init__(
        self,
        backend_names: Sequence[str],
        layout_kernels_by_backend: Dict[str, Sequence[str]],
        num_features: int,
        alpha: float = 1.0,
        ridge: float = 1e-3,
        fallback_backend: str | None = None,
    ):
        self.backend_names = list(backend_names)
        self.fallback_backend = fallback_backend or self.backend_names[0]
        self.num_features = int(num_features)
        self.backend_model = LinearUCB(self.backend_names, self.num_features, alpha=alpha, ridge=ridge)
        self.layout_models: Dict[str, LinearUCB] = {}

        for backend, kernels in layout_kernels_by_backend.items():
            self.layout_models[backend] = LinearUCB(list(kernels), self.num_features, alpha=alpha, ridge=ridge)

    def select_backend(self, context: Union[np.ndarray, Sequence[float]]) -> Tuple[str, float]:
        return self.backend_model.choose_kernel(self.backend_names, context)

    def select_kernel(self, context: Union[np.ndarray, Sequence[float]], backend: str | None = None):
        if backend is None:
            backend, _ = self.select_backend(context)

        if backend not in self.layout_models:
            return backend, None, 0.0

        kernels = list(self.layout_models[backend].models.keys())
        if not kernels:
            return backend, None, 0.0

        best_kernel, score = self.layout_models[backend].choose_kernel(kernels, context)
        return backend, best_kernel, float(score)

    def update(self, context, backend: str, kernel: str, reward: float) -> None:
        self.backend_model.update(backend, context, float(reward))
        if backend in self.layout_models:
            self.layout_models[backend].update(kernel, context, float(reward))

    def select(self, context):
        backend, _ = self.select_backend(context)
        backend_name, kernel_name, score = self.select_kernel(context, backend=backend)
        return backend_name, kernel_name, float(score)


# ------------------------------------------------------------
# compatibility helpers for the legacy profiling scripts
# ------------------------------------------------------------


def _load_runtime_modules():
    sys.path.insert(0, "/home/fakeheadset/Projects/EulerEasel/Src/include")
    try:
        import CUDAruntime as crn  # noqa: F401
        import runtime as rn  # noqa: F401
        import matrix_extractor as me  # noqa: F401
        from context import LazyFrozenContext
        from registry import launch_spmv
        return me, crn, rn, LazyFrozenContext, launch_spmv
    except Exception as exc:  # pragma: no cover - keep script compatibility
        raise RuntimeError("Runtime modules are required for profiling workflows.") from exc


def generate_ground_truth_oracle() -> None:
    me, _, _, LazyFrozenContext, launch_spmv = _load_runtime_modules()

    folder_path = Path("/home/fakeheadset/Projects/EulerEasel/Data/datasetnaked/")
    items = sorted([str(item) for item in folder_path.iterdir() if item.suffix == ".mtx"])

    str_reg = me.StrategyRegister()
    hrd = me.HardwareContext()
    strategies = str_reg.get_strategies(hrd)
    kernels = [strategy.kernel for strategy in strategies]

    oracle_records: List[dict] = []
    num_runs = 30

    for index, filename in enumerate(items, start=1):
        matrix_name = os.path.basename(filename)
        [rows, cols, nnz] = me.mat_dim(filename)
        frozen_context = LazyFrozenContext(filename, rows, cols, nnz)

        kernel_perf: Dict[str, float] = {}
        for kernel in kernels:
            runtimes: List[float] = []
            try:
                launch_spmv(kernel, frozen_context)
            except Exception:
                continue

            for _ in range(num_runs):
                result = launch_spmv(kernel, frozen_context)
                runtime = result[1] if isinstance(result, (tuple, list)) else result
                runtimes.append(float(runtime))

            if runtimes:
                kernel_perf[kernel.name] = float(np.median(runtimes))

        if not kernel_perf:
            continue

        best_kernel_name = min(kernel_perf, key=kernel_perf.get)
        best_runtime = kernel_perf[best_kernel_name]
        oracle_records.append(
            {
                matrix_name: {
                    "best_kernel": best_kernel_name,
                    "runtime": best_runtime,
                    "all_kernel_profiles": kernel_perf,
                }
            }
        )

    output_path = Path("ground_truth.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(oracle_records, handle, indent=4)

    print(f"Successfully generated oracle in: {output_path.resolve()}")


if __name__ == "__main__":
    generate_ground_truth_oracle()
        
        
            


                    
            
            
        
        


    
    
    
    
    
    
