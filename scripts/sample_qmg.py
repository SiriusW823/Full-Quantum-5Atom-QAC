"""
Quick sampling script for the Qiskit QMG generator.

Usage:
    python scripts/sample_qmg.py --samples 1000
    python scripts/sample_qmg.py --n 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import qmg` works without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qmg.generator import QiskitQMGGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="number of molecules to sample")
    parser.add_argument("--n", type=int, default=None, help="alias of --samples")
    args = parser.parse_args()

    n = args.samples if args.n is None else args.n

    gen = QiskitQMGGenerator()
    gen.sample_actions(batch_size=n)

    env = gen.env
    print(f"samples={env.metrics.samples}")
    print(f"valid_count={env.metrics.valid_count}")
    print(f"unique_valid_count={env.metrics.unique_valid_count}")
    print(f"valid_ratio={env.valid_ratio:.4f}")
    print(f"unique_ratio={env.unique_ratio:.4f}")
    print(f"target_metric={env.target_metric:.6f}")

    uniques = sorted(env.seen_smiles)[:10]
    print("Top unique SMILES (up to 10):")
    for s in uniques:
        print(s)


if __name__ == "__main__":
    main()
