from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qiskit
import qiskit_aer
from qiskit import transpile
from qiskit_aer import AerSimulator

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import FiveAtomMolEnv  # noqa: E402
from qmg.sqmg_generator import SQMGQiskitGenerator  # noqa: E402

try:  # optional CUDA-Q backend
    from qmg.cudaq_generator import CudaQMGGenerator  # type: ignore
except Exception:  # pragma: no cover
    CudaQMGGenerator = None
from qrl.a2c import A2CConfig, a2c_step, build_state  # noqa: E402
from qrl.actor import CudaQQuantumActor, QiskitQuantumActor  # noqa: E402
from qrl.critic import CudaQQuantumCritic, QiskitQuantumCritic  # noqa: E402


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
    if device in ("cuda-cpu", "cuda-gpu"):
        device = "gpu" if device.endswith("gpu") else "cpu"
    if device == "cpu":
        return "cpu", 0
    if device == "gpu":
        if gpus <= 0:
            gpus = detect_num_gpus()
        return ("gpu" if gpus > 0 else "cpu"), gpus
    if gpus <= 0:
        gpus = detect_num_gpus()
    return ("gpu" if gpus > 0 else "cpu"), gpus


def print_aer_info(selected_device: str) -> None:
    try:
        available = AerSimulator().available_devices()
    except Exception:
        available = []
    print(f"[info] qiskit={qiskit.__version__} qiskit_aer={qiskit_aer.__version__}")
    print(f"[info] AerSimulator.available_devices()={available}")
    print(f"[info] selected_device={selected_device}")


def create_aer_backend(seed: int | None, device: str, gpus: int) -> tuple[AerSimulator, bool]:
    if device != "gpu":
        return AerSimulator(seed_simulator=seed), False
    try:
        available = AerSimulator().available_devices()
    except Exception:
        available = []
    if "GPU" not in available:
        print("[warn] Aer GPU not available, falling back to CPU.")
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


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _snapshot_env(env: FiveAtomMolEnv) -> dict:
    return {
        "samples": env.metrics.samples,
        "valid_count": env.metrics.valid_count,
        "unique_valid_count": env.metrics.unique_valid_count,
        "raw_valid_count": env.metrics.raw_valid_count,
        "raw_unique_valid_count": env.metrics.raw_unique_valid_count,
        "seen_smiles": set(env.seen_smiles),
        "seen_smiles_raw": set(env.seen_smiles_raw),
    }


def _restore_env(env: FiveAtomMolEnv, snap: dict) -> None:
    env.metrics.samples = int(snap["samples"])
    env.metrics.valid_count = int(snap["valid_count"])
    env.metrics.unique_valid_count = int(snap["unique_valid_count"])
    env.metrics.raw_valid_count = int(snap["raw_valid_count"])
    env.metrics.raw_unique_valid_count = int(snap["raw_unique_valid_count"])
    env.samples = env.metrics.samples
    env.valid_count = env.metrics.valid_count
    env.unique_valid_count = env.metrics.unique_valid_count
    env.raw_valid_count = env.metrics.raw_valid_count
    env.raw_unique_valid_count = env.metrics.raw_unique_valid_count
    env.seen_smiles = set(snap["seen_smiles"])
    env.seen_smiles_raw = set(snap["seen_smiles_raw"])


def _eval_batch(batch, repair_bonds: bool) -> dict:
    eval_env = FiveAtomMolEnv(repair_bonds=repair_bonds)
    for atoms, bonds in zip(batch.atoms, batch.bonds):
        eval_env.build_smiles_from_actions(atoms, bonds)
    return eval_env.stats()


def _plot_eval(eval_rows: List[dict], out_dir: Path, warm_start_repair: int) -> None:
    if not eval_rows:
        return
    episodes = [row["episode"] for row in eval_rows]
    reward_raw = [row["reward_raw_pdf_eval"] for row in eval_rows]
    reward_rep = [row["reward_pdf_eval"] for row in eval_rows]
    validity_raw = [row["validity_raw_pdf_eval"] for row in eval_rows]
    validity_rep = [row["validity_pdf_eval"] for row in eval_rows]
    uniq_raw = [row["uniqueness_raw_pdf_eval"] for row in eval_rows]
    uniq_rep = [row["uniqueness_pdf_eval"] for row in eval_rows]

    best_raw = []
    best = 0.0
    for r in reward_raw:
        best = max(best, r)
        best_raw.append(best)

    if warm_start_repair > 0:
        vline = warm_start_repair
    else:
        vline = None

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, reward_raw, label="reward_raw_pdf_eval")
    plt.plot(episodes, reward_rep, label="reward_pdf_eval")
    plt.plot(episodes, best_raw, label="best_raw_so_far")
    if vline is not None:
        plt.axvline(vline, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Eval Reward (PDF)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_eval.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, validity_raw, label="validity_raw_pdf_eval")
    plt.plot(episodes, validity_rep, label="validity_pdf_eval")
    if vline is not None:
        plt.axvline(vline, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Episode")
    plt.ylabel("Validity")
    plt.title("Eval Validity (PDF)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validity_eval.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, uniq_raw, label="uniqueness_raw_pdf_eval")
    plt.plot(episodes, uniq_rep, label="uniqueness_pdf_eval")
    if vline is not None:
        plt.axvline(vline, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Episode")
    plt.ylabel("Uniqueness")
    plt.title("Eval Uniqueness (PDF)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "uniqueness_eval.png", dpi=150)
    plt.close()


def run_one_train(
    episodes: int = 300,
    batch_size: int = 256,
    out_dir: str | Path = "runs/dgx_run",
    device: str = "auto",
    backend: str = "qiskit",
    gpus: int = 0,
    seed: int = 123,
    atom_layers: int = 2,
    bond_layers: int = 1,
    repair_bonds: bool = False,
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
    k_batches: int = 2,
    beta_novelty: float = 0.0,
    lambda_repeat: float = 0.0,
    ent_coef: float = 0.01,
    reward_floor: float = 0.0,
    reward_clip_low: float = 0.0,
    reward_clip_high: float = 1.0,
    sigma_min: float = 0.1,
    sigma_max: float = 1.0,
    sigma_boost: float = 1.25,
    sigma_decay: float = 0.997,
    patience: int = 50,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    log_every: int = 10,
    track_best: bool = False,
    eval_every: int = 50,
    eval_shots: int = 2000,
    warm_start_repair: int = 0,
    adaptive_exploration: bool = True,
    adapt_threshold: float = 0.01,
    adapt_window: int = 5,
) -> List[Dict[str, float]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path / "metrics.csv"
    plot_path = out_path / "reward.png"
    eval_rows: List[dict] = []
    strict_eval_rewards: List[float] = []

    _set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    resolved_device, detected_gpus = resolve_device(device, gpus)
    if backend == "cudaq":
        if CudaQMGGenerator is None:
            raise RuntimeError("CUDA-Q backend requested but cudaq is not installed.")
        qmg = CudaQMGGenerator(
            atom_layers=atom_layers,
            bond_layers=bond_layers,
            repair_bonds=repair_bonds,
            device=resolved_device,
            seed=seed,
        )
        try:
            import cudaq  # noqa: WPS433

            print(
                f"[info] cudaq_version={getattr(cudaq, '__version__', 'unknown')} "
                f"target={getattr(qmg, 'target', 'unknown')} device={resolved_device}"
            )
        except Exception:
            print("[warn] cudaq backend selected but unable to query cudaq details.")
    else:
        qmg = SQMGQiskitGenerator(
            atom_layers=atom_layers,
            bond_layers=bond_layers,
            repair_bonds=repair_bonds,
            seed=seed,
        )

    if backend == "qiskit":
        print_aer_info(resolved_device)
        aer_backend, using_gpu = create_aer_backend(seed, resolved_device, detected_gpus)
        configure_generator_backend(qmg, aer_backend)
        if resolved_device == "gpu" and using_gpu:
            print(f"[info] Using Aer GPU backend (gpus={max(detected_gpus, 1)})")
        else:
            print("[info] Using Aer CPU backend")

    state_dim = build_state(qmg).size
    if backend == "cudaq":
        actor = CudaQQuantumActor(
            state_dim=state_dim,
            n_qubits=actor_qubits,
            n_layers=actor_layers,
            action_dim=action_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            seed=seed,
        )
        critic = CudaQQuantumCritic(
            state_dim=state_dim,
            n_qubits=critic_qubits,
            n_layers=critic_layers,
            seed=seed + 1,
        )
    else:
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
        track_best=track_best,
    )

    rows: List[Dict[str, float]] = []
    fieldnames = [
        "episode",
        "phase",
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

    eval_path = out_path / "eval.csv"
    if not eval_path.exists():
        eval_path.write_text(
            "episode,phase,reward_pdf_eval,reward_raw_pdf_eval,validity_pdf_eval,uniqueness_pdf_eval,"
            "validity_raw_pdf_eval,uniqueness_raw_pdf_eval,sigma_max,k_batches,patience,eval_seed\n"
        )
    best_json_path = out_path / "best_eval.json"

    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        prev_phase = None
        for ep in range(1, episodes + 1):
            phase = "repair" if ep <= warm_start_repair else "strict"
            qmg.env.repair_bonds = phase == "repair"
            if track_best and prev_phase == "repair" and phase == "strict":
                cfg.best_reward_pdf = 0.0
                cfg.best_weights = None
                print("[info] Switched to strict phase; reset best checkpoint tracking.")
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
                "phase": phase,
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
            stats = qmg.env.stats()
            if ep % log_every == 0 or ep == 1:
                print(
                    f"raw_vs_repair: raw_reward={stats['reward_raw_pdf']:.6f} "
                    f"repaired_reward={stats['reward_pdf']:.6f}"
                )
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

            if eval_every > 0 and ep % eval_every == 0:
                weights_snapshot = qmg.get_weights()
                env_snap = _snapshot_env(qmg.env)
                eval_batch = qmg.sample_actions(batch_size=eval_shots)
                eval_stats = _eval_batch(eval_batch, repair_bonds=(phase == "repair"))
                _restore_env(qmg.env, env_snap)

                reward_pdf_eval = eval_stats["reward_pdf"]
                reward_raw_pdf_eval = eval_stats["reward_raw_pdf"]
                validity_pdf_eval = eval_stats["validity_pdf"]
                uniqueness_pdf_eval = eval_stats["uniqueness_pdf"]
                validity_raw_pdf_eval = eval_stats["validity_raw_pdf"]
                uniqueness_raw_pdf_eval = eval_stats["uniqueness_raw_pdf"]
                print(
                    f"[eval {ep}] reward_pdf_eval={reward_pdf_eval:.6f} "
                    f"reward_raw_pdf_eval={reward_raw_pdf_eval:.6f} "
                    f"validity_pdf_eval={validity_pdf_eval:.4f} "
                    f"uniqueness_pdf_eval={uniqueness_pdf_eval:.4f} "
                    f"phase={phase}"
                )

                if adaptive_exploration and phase == "strict":
                    strict_eval_rewards.append(float(reward_raw_pdf_eval))
                    if len(strict_eval_rewards) > adapt_window:
                        strict_eval_rewards = strict_eval_rewards[-adapt_window:]
                    if len(strict_eval_rewards) == adapt_window and all(
                        r < adapt_threshold for r in strict_eval_rewards
                    ):
                        cfg.sigma_max = min(cfg.sigma_max * 1.2, 2.0)
                        cfg.patience = min(cfg.patience + 20, 200)
                        cfg.k_batches = min(cfg.k_batches + 1, 8)
                        actor.sigma_max = cfg.sigma_max
                        if cfg.sigma_current is not None:
                            cfg.sigma_current = min(cfg.sigma_current, cfg.sigma_max)
                        print(
                            "[adapt] reward below threshold: "
                            f"sigma_max={cfg.sigma_max:.3f} "
                            f"k_batches={cfg.k_batches} "
                            f"patience={cfg.patience}"
                        )

                eval_row = {
                    "episode": ep,
                    "phase": phase,
                    "reward_pdf_eval": reward_pdf_eval,
                    "reward_raw_pdf_eval": reward_raw_pdf_eval,
                    "validity_pdf_eval": validity_pdf_eval,
                    "uniqueness_pdf_eval": uniqueness_pdf_eval,
                    "validity_raw_pdf_eval": validity_raw_pdf_eval,
                    "uniqueness_raw_pdf_eval": uniqueness_raw_pdf_eval,
                    "sigma_max": cfg.sigma_max,
                    "k_batches": cfg.k_batches,
                    "patience": cfg.patience,
                    "eval_seed": seed,
                }
                eval_rows.append(eval_row)
                with eval_path.open("a", newline="") as handle:
                    handle.write(
                        f"{ep},{phase},{reward_pdf_eval:.6f},{reward_raw_pdf_eval:.6f},"
                        f"{validity_pdf_eval:.6f},{uniqueness_pdf_eval:.6f},"
                        f"{validity_raw_pdf_eval:.6f},{uniqueness_raw_pdf_eval:.6f},"
                        f"{cfg.sigma_max:.6f},{cfg.k_batches},{cfg.patience},{seed}\n"
                    )

                if track_best:
                    metric_for_best = reward_raw_pdf_eval if phase == "strict" else reward_pdf_eval
                    if cfg.best_weights is None or metric_for_best > cfg.best_reward_pdf:
                        cfg.best_reward_pdf = metric_for_best
                        cfg.best_weights = weights_snapshot.copy()
                        best_json = {
                            "episode": ep,
                            "phase": phase,
                            "reward_pdf_eval": reward_pdf_eval,
                            "reward_raw_pdf_eval": reward_raw_pdf_eval,
                            "validity_pdf_eval": validity_pdf_eval,
                            "uniqueness_pdf_eval": uniqueness_pdf_eval,
                            "validity_raw_pdf_eval": validity_raw_pdf_eval,
                            "uniqueness_raw_pdf_eval": uniqueness_raw_pdf_eval,
                            "metric_for_best": metric_for_best,
                            "sigma_max": cfg.sigma_max,
                            "k_batches": cfg.k_batches,
                            "patience": cfg.patience,
                        }
                        best_json_path.write_text(json.dumps(best_json, indent=2))
                        np.save(out_path / "best_weights.npy", cfg.best_weights)

                qmg.set_weights(weights_snapshot)

            prev_phase = phase

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

    _plot_eval(eval_rows, out_path, warm_start_repair)

    tail = rewards[-50:] if len(rewards) >= 50 else rewards
    mean_reward = float(np.mean(tail)) if tail else 0.0
    print(f"Final summary: mean reward (last {len(tail)} episodes) = {mean_reward:.6f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved reward plot to {plot_path}")

    if track_best and cfg.best_weights is not None:
        best_path = out_path / "best_weights.npy"
        np.save(best_path, cfg.best_weights)
        print(f"Saved best weights to {best_path}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=str, default="runs/dgx_run")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "cuda-cpu", "cuda-gpu"],
        default="auto",
    )
    parser.add_argument("--backend", choices=["qiskit", "cudaq"], default="qiskit")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atom-layers", type=int, default=2)
    parser.add_argument("--bond-layers", type=int, default=1)
    parser.add_argument("--repair-bonds", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--k-batches", type=int, default=2)
    parser.add_argument("--beta-novelty", type=float, default=0.0)
    parser.add_argument("--lambda-repeat", type=float, default=0.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-floor", type=float, default=0.0)
    parser.add_argument("--reward-clip-low", type=float, default=0.0)
    parser.add_argument("--reward-clip-high", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-boost", type=float, default=1.25)
    parser.add_argument("--sigma-decay", type=float, default=0.997)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--track-best", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-shots", type=int, default=2000)
    parser.add_argument("--warm-start-repair", type=int, default=0)
    parser.add_argument(
        "--adaptive-exploration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adapt-threshold", type=float, default=0.01)
    parser.add_argument("--adapt-window", type=int, default=5)

    args = parser.parse_args()
    run_one_train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        out_dir=args.out,
        device=args.device,
        backend=args.backend,
        gpus=args.gpus,
        seed=args.seed,
        atom_layers=args.atom_layers,
        bond_layers=args.bond_layers,
        repair_bonds=args.repair_bonds,
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
        track_best=args.track_best,
        eval_every=args.eval_every,
        eval_shots=args.eval_shots,
        warm_start_repair=args.warm_start_repair,
        adaptive_exploration=args.adaptive_exploration,
        adapt_threshold=args.adapt_threshold,
        adapt_window=args.adapt_window,
    )


if __name__ == "__main__":
    main()
