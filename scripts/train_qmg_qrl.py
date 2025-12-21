"""
Joint training of Qiskit QMG generator and QRL helper.

Usage:
    python -m scripts.train_qmg_qrl --steps 2000 --batch-size 64
    python -m scripts.train_qmg_qrl --algo a2c --steps 200 --batch-size 256
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
from qmg.sqmg_generator import SQMGQiskitGenerator  # noqa: E402
from qrl.a2c import A2CConfig, a2c_step, build_state  # noqa: E402
from qrl.actor import QiskitQuantumActor  # noqa: E402
from qrl.critic import QiskitQuantumCritic  # noqa: E402
from qrl.helper import QiskitQRLHelper  # noqa: E402


def compute_rewards(
    batch, qrl_helper: QiskitQRLHelper, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
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


def run_helper(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    qmg = QiskitQMGGenerator(seed=args.seed)
    qrl = QiskitQRLHelper(seed=args.seed)

    start_noise = 0.3
    end_noise = 0.05

    for step in range(1, args.steps + 1):
        noise_std = start_noise - (start_noise - end_noise) * (step - 1) / max(1, args.steps - 1)

        w_base = qmg.get_weights()
        delta = rng.choice([-1.0, 1.0], size=w_base.shape)
        pert_scale = args.qmg_eps

        w_plus = w_base + pert_scale * delta + rng.normal(0.0, noise_std, size=w_base.shape)
        qmg.set_weights(w_plus)
        batch_plus = qmg.sample_actions(batch_size=args.batch_size)
        rewards_plus, targets_plus = compute_rewards(batch_plus, qrl, alpha=args.alpha)
        mean_plus = float(np.mean(rewards_plus))

        qrl.train_step(batch_plus.smiles, targets_plus, lr=args.qrl_lr, spsa_eps=args.qrl_eps)

        w_minus = w_base - pert_scale * delta + rng.normal(0.0, noise_std, size=w_base.shape)
        qmg.set_weights(w_minus)
        batch_minus = qmg.sample_actions(batch_size=args.batch_size)
        rewards_minus, _ = compute_rewards(batch_minus, qrl, alpha=args.alpha)
        mean_minus = float(np.mean(rewards_minus))

        ghat = (mean_plus - mean_minus) / (2.0 * pert_scale) * delta
        w_updated = w_base + args.qmg_lr * ghat
        qmg.set_weights(w_updated)

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


def run_a2c(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    qmg = SQMGQiskitGenerator(
        atom_layers=args.atom_layers,
        bond_layers=args.bond_layers,
        seed=args.seed,
    )

    state_dim = build_state(qmg).size
    actor = QiskitQuantumActor(
        state_dim=state_dim,
        n_qubits=args.actor_qubits,
        n_layers=args.actor_layers,
        action_dim=args.action_dim,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        seed=args.seed,
    )
    critic = QiskitQuantumCritic(
        state_dim=state_dim,
        n_qubits=args.critic_qubits,
        n_layers=args.critic_layers,
        seed=args.seed + 1,
    )

    proj = rng.normal(0.0, 1.0, size=(qmg.num_weights, args.action_dim))
    proj /= np.sqrt(args.action_dim)

    cfg = A2CConfig(
        action_dim=args.action_dim,
        lr_theta=args.lr_theta,
        actor_a=args.actor_a,
        actor_c=args.actor_c,
        critic_a=args.critic_a,
        critic_c=args.critic_c,
        k_batches=args.k_batches,
        beta_novelty=args.beta_novelty,
        lambda_repeat=args.lambda_repeat,
        ent_coef=args.ent_coef,
        reward_floor=args.reward_floor,
        reward_clip_low=args.reward_clip_low,
        reward_clip_high=args.reward_clip_high,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_boost=args.sigma_boost,
        sigma_decay=args.sigma_decay,
        patience=args.patience,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
    )

    for step in range(1, args.steps + 1):
        state = build_state(qmg)
        result = a2c_step(
            gen=qmg,
            actor=actor,
            critic=critic,
            proj=proj,
            rng=rng,
            batch_size=args.batch_size,
            cfg=cfg,
            state=state,
        )

        if step % args.log_every == 0 or step == 1:
            print(f"[step {step}]")
            print(
                f"samples={qmg.env.samples} valid={qmg.env.valid_count} unique={qmg.env.unique_valid_count}"
            )
            print(
                f"reward={result['reward']:.6f} reward_avg={result['reward_avg']:.6f} "
                f"reward_main={result['reward_main']:.6f} repeat_penalty={result['repeat_penalty']:.6f}"
            )
            print(
                f"step_ratios: validity_step={result['validity_step']:.4f} "
                f"uniqueness_step={result['uniqueness_step']:.4f} "
                f"score_pdf_step={result['score_pdf_step']:.6f} "
                f"novelty_step={result['novelty_step']:.6f} "
                f"repeat_step={result['repeat_step']:.6f}"
            )
            print(
                f"step_deltas: ds={result['ds']:.0f} dv={result['dv']:.0f} "
                f"unique_in_batch={result['unique_valid_in_batch']:.0f} "
                f"novel_in_batch={result['novel_valid_in_batch']:.0f} "
                f"sigma={result['sigma']:.4f} entropy={result['entropy']:.6f}"
            )
            print(
                f"value={result['value']:.6f} advantage={result['advantage']:.6f} "
                f"actor_loss={result['actor_loss']:.6f} critic_loss={result['critic_loss']:.6f}"
            )
            stats_pdf = qmg.env.stats()
            print(
                f"pdf_cumulative: validity_pdf={stats_pdf['validity_pdf']:.4f} "
                f"uniqueness_pdf={stats_pdf['uniqueness_pdf']:.4f} "
                f"target_metric_pdf={stats_pdf['target_metric_pdf']:.6f}"
            )
            uniques = sorted(qmg.env.seen_smiles)[:10]
            print("Top unique SMILES (up to 10):")
            for smi in uniques:
                print(smi)
            print("-" * 60)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_batch = qmg.sample_actions(batch_size=args.eval_batch_size)
            eval_ds = args.eval_batch_size
            eval_du = int(sum(1 for u in getattr(eval_batch, "uniques", []) if u))
            eval_composite = eval_du / eval_ds if eval_ds else 0.0
            stats = qmg.env.stats()
            print(
                f"[eval step {step}] samples={stats['samples']} eval_composite={eval_composite:.6f}"
            )
            print("-" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["helper", "a2c"], default="helper")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on qrl score for unique")
    parser.add_argument("--qmg-lr", type=float, default=0.05)
    parser.add_argument("--qmg-eps", type=float, default=0.1)
    parser.add_argument("--qrl-lr", type=float, default=0.05)
    parser.add_argument("--qrl-eps", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--lr-theta", type=float, default=0.03)
    parser.add_argument("--actor-a", type=float, default=0.05)
    parser.add_argument("--actor-c", type=float, default=0.01)
    parser.add_argument("--critic-a", type=float, default=0.05)
    parser.add_argument("--critic-c", type=float, default=0.01)
    parser.add_argument("--k-batches", type=int, default=3)
    parser.add_argument("--beta-novelty", type=float, default=0.05)
    parser.add_argument("--lambda-repeat", type=float, default=0.10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-floor", type=float, default=-0.01)
    parser.add_argument("--reward-clip-low", type=float, default=-0.05)
    parser.add_argument("--reward-clip-high", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.50)
    parser.add_argument("--sigma-boost", type=float, default=1.5)
    parser.add_argument("--sigma-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    parser.add_argument("--actor-qubits", type=int, default=8)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-qubits", type=int, default=8)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--atom-layers", type=int, default=2)
    parser.add_argument("--bond-layers", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=2000)

    args = parser.parse_args()

    if args.algo == "a2c":
        run_a2c(args)
    else:
        run_helper(args)


if __name__ == "__main__":
    main()
