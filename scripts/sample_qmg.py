"""scripts/sample_qmg.py

Sample molecules from QMG.

Examples:
  python -m scripts.sample_qmg --mode sqmg --n 2000
  python -m scripts.sample_qmg --mode factorized --n 2000
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
from qmg.sqmg_generator import SQMGQiskitGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--mode", choices=["sqmg", "factorized"], default="sqmg")
    parser.add_argument("--shots", type=int, default=256, help="shots for SQMG only")
    parser.add_argument("--atom-layers", type=int, default=2)
    parser.add_argument("--bond-layers", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "sqmg":
        gen = SQMGQiskitGenerator(
            atom_layers=args.atom_layers,
            bond_layers=args.bond_layers,
            shots=args.shots,
        )
    else:
        gen = QiskitQMGGenerator()

    gen.sample_actions(batch_size=args.n)

    env = gen.env
    qc = getattr(gen, "base_circuit", None)
    if qc is not None:
        print(f"circuit: n_qubits={qc.num_qubits} n_clbits={qc.num_clbits}")
    if args.mode == "sqmg":
        print(f"shots={args.shots} atom_layers={args.atom_layers} bond_layers={args.bond_layers}")

    s = env.stats() if hasattr(env, "stats") else {}
    if s:
        print(f"samples={s['samples']}")
        print(f"valid_count={s['valid_count']}")
        print(f"unique_valid_count={s['unique_valid_count']}")
        print(f"valid_ratio={s['valid_ratio']:.4f}")
        print(f"unique_ratio={s['unique_ratio']:.4f}")
        print(f"target_metric={s['target_metric']:.6f}")

    uniques = sorted(env.seen_smiles)[:10]
    print("Top unique SMILES (up to 10):")
    for smi in uniques:
        print(smi)


if __name__ == "__main__":
    main()
