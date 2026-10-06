"""Run contextual-policy replay and optional kernel parameter tuning.

Examples (run from the repository root):
    python Testing/algorithm_tuning_experiment.py
    python Testing/algorithm_tuning_experiment.py --tune-matrix dataset/bcsstk18.mtx

The implementation lives in :mod:`Testing.algorithm_tuning`; this file remains
the stable command-line entry point used by existing scripts and worker processes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Testing.algorithm_tuning.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
