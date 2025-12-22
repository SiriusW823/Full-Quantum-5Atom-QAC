from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qmg.sqmg_generator import SQMGQiskitGenerator  # noqa: E402
from qrl.a2c import A2CConfig, a2c_step, build_state  # noqa: E402
from qrl.actor import QiskitQuantumActor  # noqa: E402
from qrl.critic import QiskitQuantumCritic  # noqa: E402


def detect_num_gpus() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        ids = [x for x in env.split(",") if x.strip() != ""]
        return len(ids)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 0
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU"))


def resolve_device(device: str, gpus: int) -> Tuple[str, int]:
    device = device.lower()
    if device == "cpu":
        return "cpu", 0
    if device == "gpu":
        if gpus <= 0:
            gpus = detect_num_gpus()
        return ("gpu" if gpus > 0 else "cpu"), gpus
    if gpus <= 0:
        gpus = detect_num_gpus()
    return ("gpu" if gpus > 0 else "cpu"), gpus


def create_aer_backend(seed: int | None, device: str, gpus: int) -> tuple[AerSimulator, bool]:
    if device != "gpu":
        return AerSimulator(seed_simulator=seed), False
    try:
        backend = AerSimulator(device="GPU", seed_simulator=seed)
        if gpus > 1:
            try:
                backend.set_options(batched_shots_gpu=gpus)
            except Exception:
                pass
        return backend, True
    except Exception as exc:
        print(f"[warn] GPU backend unavailable, falling back to CPU: {exc}")
        return AerSimulator(seed_simulator=seed), False


def configure_generator_backend(gen: SQMGQiskitGenerator, backend: AerSimulator) -> None:
    gen.backend = backend
    gen._compiled = transpile(gen.base_circuit, backend, optimization_level=1)


def run_one_train(
    episodes: int = 300,
    batch_size: int = 256,
    out_dir: str | Path = "runs/dgx_run",
    device: str = "auto",
    gpus: int = 0,
    seed: int = 123,
    atom_layers: int = 2,
    bond_layers: int = 1,
    actor_qubits: int = 8,
    actor_layers: int = 2,
    critic_qubits: int = 8,
    critic_layers: int = 2,
    action_dim: int = 16,
    lr_theta: float = 0.03,
    actor_a: float = 0.05,
    actor_c: float = 0.01,
    critic_a: float = 0.05,
    critic_c: float = 0.01,
    k_batches: int = 1,
    beta_novelty: float = 0.0,
    lambda_repeat: float = 0.0,
    ent_coef: float = 0.01,
    reward_floor: float = 0.0,
    reward_clip_low: float = -0.05,
    reward_clip_high: float = 1.0,
    sigma_min: float = 0.05,
    sigma_max: float = 0.50,
    sigma_boost: float = 1.25,
    sigma_decay: float = 0.995,
    patience: int = 25,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    log_every: int = 10,
) -> List[Dict[str, float]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path / "metrics.csv"
    plot_path = out_path / "reward.png"

    rng = np.random.default_rng(seed)
    qmg = SQMGQiskitGenerator(atom_layers=atom_layers, bond_layers=bond_layers, seed=seed)

    resolved_device, detected_gpus = resolve_device(device, gpus)
    backend, using_gpu = create_aer_backend(seed, resolved_device, detected_gpus)
    configure_generator_backend(qmg, backend)
    if resolved_device == "gpu" and using_gpu:
        print(f"[info] Using Aer GPU backend (gpus={max(detected_gpus, 1)})")
    else:
        print("[info] Using Aer CPU backend")

    state_dim = build_state(qmg).size
    actor = QiskitQuantumActor(
        state_dim=state_dim,
        n_qubits=actor_qubits,
        n_layers=actor_layers,
        action_dim=action_dim,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        seed=seed,
    )
    critic = QiskitQuantumCritic(
        state_dim=state_dim,
        n_qubits=critic_qubits,
        n_layers=critic_layers,
        seed=seed + 1,
    )

    proj = rng.normal(0.0, 1.0, size=(qmg.num_weights, action_dim))
    proj /= np.sqrt(action_dim)

    cfg = A2CConfig(
        action_dim=action_dim,
        lr_theta=lr_theta,
        actor_a=actor_a,
        actor_c=actor_c,
        critic_a=critic_a,
        critic_c=critic_c,
        k_batches=k_batches,
        beta_novelty=beta_novelty,
        lambda_repeat=lambda_repeat,
        ent_coef=ent_coef,
        reward_floor=reward_floor,
        reward_clip_low=reward_clip_low,
        reward_clip_high=reward_clip_high,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_boost=sigma_boost,
        sigma_decay=sigma_decay,
        patience=patience,
        spsa_alpha=spsa_alpha,
        spsa_gamma=spsa_gamma,
    )

    rows: List[Dict[str, float]] = []
    fieldnames = [
        "episode",
        "reward_step",
        "validity_step",
        "uniqueness_step",
        "valid_count_step",
        "unique_valid_count_step",
        "batch_size",
        "sigma",
        "entropy",
        "reward_used",
    ]

    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            state = build_state(qmg)
            result = a2c_step(
                gen=qmg,
                actor=actor,
                critic=critic,
                proj=proj,
                rng=rng,
                batch_size=batch_size,
                cfg=cfg,
                state=state,
            )

            reward_step = float(result.get("reward_step", result["validity_step"] * result["uniqueness_step"]))
            row = {
                "episode": ep,
                "reward_step": reward_step,
                "validity_step": float(result["validity_step"]),
                "uniqueness_step": float(result["uniqueness_step"]),
                "valid_count_step": int(result["dv"]),
                "unique_valid_count_step": int(result["du"]),
                "batch_size": int(result["ds"]),
                "sigma": float(result.get("sigma", 0.0)),
                "entropy": float(result.get("entropy", 0.0)),
                "reward_used": float(result["reward"]),
            }
            writer.writerow(row)
            rows.append(row)

            if ep % log_every == 0 or ep == 1:
                print(
                    f"[ep {ep}] reward_step={row['reward_step']:.6f} "
                    f"validity_step={row['validity_step']:.4f} "
                    f"uniqueness_step={row['uniqueness_step']:.4f} "
                    f"dv={row['valid_count_step']} du={row['unique_valid_count_step']} "
                    f"sigma={row['sigma']:.4f}"
                )

    rewards = [r["reward_step"] for r in rows]
    if rewards:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(rewards) + 1), rewards, label="reward_step")
        plt.xlabel("Episode")
        plt.ylabel("Reward (validity_step * uniqueness_step)")
        plt.title("QMG+QRL Training Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    tail = rewards[-50:] if len(rewards) >= 50 else rewards
    mean_reward = float(np.mean(tail)) if tail else 0.0
    print(f"Final summary: mean reward (last {len(tail)} episodes) = {mean_reward:.6f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved reward plot to {plot_path}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=str, default="runs/dgx_run")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atom-layers", type=int, default=2)
    parser.add_argument("--bond-layers", type=int, default=1)
    parser.add_argument("--actor-qubits", type=int, default=8)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-qubits", type=int, default=8)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--lr-theta", type=float, default=0.03)
    parser.add_argument("--actor-a", type=float, default=0.05)
    parser.add_argument("--actor-c", type=float, default=0.01)
    parser.add_argument("--critic-a", type=float, default=0.05)
    parser.add_argument("--critic-c", type=float, default=0.01)
    parser.add_argument("--k-batches", type=int, default=1)
    parser.add_argument("--beta-novelty", type=float, default=0.0)
    parser.add_argument("--lambda-repeat", type=float, default=0.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-floor", type=float, default=0.0)
    parser.add_argument("--reward-clip-low", type=float, default=-0.05)
    parser.add_argument("--reward-clip-high", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.50)
    parser.add_argument("--sigma-boost", type=float, default=1.25)
    parser.add_argument("--sigma-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()
    run_one_train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        out_dir=args.out,
        device=args.device,
        gpus=args.gpus,
        seed=args.seed,
        atom_layers=args.atom_layers,
        bond_layers=args.bond_layers,
        actor_qubits=args.actor_qubits,
        actor_layers=args.actor_layers,
        critic_qubits=args.critic_qubits,
        critic_layers=args.critic_layers,
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
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
