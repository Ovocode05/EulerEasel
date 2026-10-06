import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Src" / "include"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model.linTS import FeatureNormalizer, LinearUCB


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
