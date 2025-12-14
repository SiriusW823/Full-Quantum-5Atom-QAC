"""
Quick sampling script for the Qiskit QMG generator.

Usage:
    python scripts/sample_qmg.py --samples 1000
"""

from __future__ import annotations

import argparse

from qmg.generator import QiskitQMGGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="number of molecules to sample")
    args = parser.parse_args()

    gen = QiskitQMGGenerator()
    batch = gen.sample_actions(batch_size=args.samples)

    env = gen.env
    print(f"samples={env.metrics.samples}")
    print(f"valid_count={env.metrics.valid_count}")
    print(f"unique_valid_count={env.metrics.unique_valid_count}")
    print(f"valid_ratio={env.valid_ratio:.4f}")
    print(f"unique_ratio={env.unique_ratio:.4f}")
    print(f"target_metric={env.target_metric:.6f}")

    # show up to 10 unique SMILES
    uniques = list(env.seen_smiles)[:10]
    print("Top unique SMILES (up to 10):")
    for s in uniques:
        print(s)


if __name__ == "__main__":
    main()
