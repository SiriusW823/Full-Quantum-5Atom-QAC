"""
Joint training of Qiskit QMG generator and QRL helper.

Usage:
    python -m scripts.train_qmg_qrl --steps 2000 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qmg.generator import QiskitQMGGenerator  # noqa: E402
from qrl.helper import QiskitQRLHelper  # noqa: E402


def compute_rewards(batch, qrl_helper: QiskitQRLHelper, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    smiles = batch.smiles
    valids = batch.valids
    uniques = getattr(batch, "uniques", [False] * len(smiles))
    scores = qrl_helper.score([s if s else "" for s in smiles])

    rewards = np.zeros(len(smiles), dtype=float)
    targets = np.zeros(len(smiles), dtype=float)
    for i, smi in enumerate(smiles):
        if not valids[i]:
            rewards[i] = -0.2
            targets[i] = 0.0
            continue
        if uniques[i]:
            rewards[i] = 1.0 + alpha * scores[i]
            targets[i] = 1.0
        else:
            rewards[i] = -0.05
            targets[i] = 0.0
    return rewards, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on qrl score for unique")
    parser.add_argument("--qmg-lr", type=float, default=0.05)
    parser.add_argument("--qmg-eps", type=float, default=0.1)
    parser.add_argument("--qrl-lr", type=float, default=0.05)
    parser.add_argument("--qrl-eps", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    qmg = QiskitQMGGenerator(seed=args.seed)
    qrl = QiskitQRLHelper(seed=args.seed)

    start_noise = 0.3
    end_noise = 0.05

    for step in range(1, args.steps + 1):
        # anneal noise
        noise_std = start_noise - (start_noise - end_noise) * (step - 1) / max(1, args.steps - 1)

        # SPSA for QMG
        w_base = qmg.get_weights()
        delta = rng.choice([-1.0, 1.0], size=w_base.shape)
        pert_scale = args.qmg_eps

        w_plus = w_base + pert_scale * delta + rng.normal(0.0, noise_std, size=w_base.shape)
        qmg.set_weights(w_plus)
        batch_plus = qmg.sample_actions(batch_size=args.batch_size)
        rewards_plus, targets_plus = compute_rewards(batch_plus, qrl, alpha=args.alpha)
        mean_plus = float(np.mean(rewards_plus))

        # train QRL on this batch
        qrl.train_step(batch_plus.smiles, targets_plus, lr=args.qrl_lr, spsa_eps=args.qrl_eps)

        w_minus = w_base - pert_scale * delta + rng.normal(0.0, noise_std, size=w_base.shape)
        qmg.set_weights(w_minus)
        batch_minus = qmg.sample_actions(batch_size=args.batch_size)
        rewards_minus, _ = compute_rewards(batch_minus, qrl, alpha=args.alpha)
        mean_minus = float(np.mean(rewards_minus))

        ghat = (mean_plus - mean_minus) / (2.0 * pert_scale) * delta
        w_updated = w_base + args.qmg_lr * ghat
        qmg.set_weights(w_updated)

        # eval on updated weights
        batch_eval = qmg.sample_actions(batch_size=args.batch_size)
        rewards_eval, targets_eval = compute_rewards(batch_eval, qrl, alpha=args.alpha)
        qrl_result = qrl.train_step(batch_eval.smiles, targets_eval, lr=args.qrl_lr, spsa_eps=args.qrl_eps)

        if step % args.log_every == 0 or step == 1:
            env = qmg.env
            print(f"[step {step}]")
            print(
                f"samples={env.samples} valid={env.valid_count} unique={env.unique_valid_count} "
                f"valid_ratio={env.valid_ratio:.4f} unique_ratio={env.unique_ratio:.4f} "
                f"target_metric={env.target_metric:.6f}"
            )
            print(
                f"mean_reward={np.mean(rewards_eval):.4f} qrl_loss={qrl_result.loss:.4f} "
                f"noise_std={noise_std:.4f}"
            )
            uniques = sorted(env.seen_smiles)[:10]
            print("Top unique SMILES (up to 10):")
            for s in uniques:
                print(s)
            print("-" * 60)


if __name__ == "__main__":
    main()
