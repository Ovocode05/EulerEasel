"""Import-path support for the compiled Python extension modules."""

import sys
from pathlib import Path

INCLUDE_DIR = Path(__file__).resolve().parent
NATIVE_MODULE_DIR = INCLUDE_DIR / "native"


def ensure_native_module_path() -> None:
    """Add the CMake extension output directory to Python's import path.

    Usage: call before importing ``matrix_extractor``, ``runtime``, or the optional
    ``CUDAruntime`` extension when those modules are built under ``include/native``.
    """
    native_path = str(NATIVE_MODULE_DIR)
    if native_path not in sys.path:
        sys.path.insert(0, native_path)
