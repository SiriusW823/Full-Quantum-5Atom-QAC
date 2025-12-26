"""
Joint training of Qiskit QMG generator and QRL helper.

Usage:
    python -m scripts.train_qmg_qrl --steps 5000 --batch-size 256
    python -m scripts.train_qmg_qrl --algo a2c --steps 5000 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

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
from qmg.generator import QiskitQMGGenerator  # noqa: E402
from qmg.sqmg_generator import SQMGQiskitGenerator  # noqa: E402

try:  # optional CUDA-Q backend
    from qmg.cudaq_generator import CudaQMGGenerator  # type: ignore
except Exception:  # pragma: no cover
    CudaQMGGenerator = None
from qrl.a2c import A2CConfig, a2c_step, build_state  # noqa: E402
from qrl.actor import CudaQQuantumActor, QiskitQuantumActor  # noqa: E402
from qrl.critic import CudaQQuantumCritic, QiskitQuantumCritic  # noqa: E402
from qrl.helper import QiskitQRLHelper  # noqa: E402

def _resolve_cudaq_device(device: str) -> str:
    device = device.lower()
    if device in ("cuda-gpu", "gpu", "nvidia"):
        return "gpu"
    if device in ("cuda-cpu", "cpu", "qpp"):
        return "cpu"
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        ids = [x for x in env.split(",") if x.strip() != ""]
        return "gpu" if ids else "cpu"
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "cpu"
    if result.returncode != 0:
        return "cpu"
    return "gpu" if any(line.strip().startswith("GPU") for line in result.stdout.splitlines()) else "cpu"


def _detect_num_gpus() -> int:
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


def _resolve_qiskit_device(device: str, gpus: int) -> tuple[str, int]:
    device = device.lower()
    if device in ("cuda-cpu", "cuda-gpu"):
        device = "gpu" if device.endswith("gpu") else "cpu"
    if device == "cpu":
        return "cpu", 0
    if device == "gpu":
        if gpus <= 0:
            gpus = _detect_num_gpus()
        return ("gpu" if gpus > 0 else "cpu"), gpus
    if gpus <= 0:
        gpus = _detect_num_gpus()
    return ("gpu" if gpus > 0 else "cpu"), gpus


def _print_aer_info(selected_device: str) -> None:
    try:
        available = AerSimulator().available_devices()
    except Exception:
        available = []
    print(f"[info] qiskit={qiskit.__version__} qiskit_aer={qiskit_aer.__version__}")
    print(f"[info] AerSimulator.available_devices()={available}")
    print(f"[info] selected_device={selected_device}")


def _create_aer_backend(seed: int | None, device: str, gpus: int) -> tuple[AerSimulator, bool]:
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


def _configure_generator_backend(gen: SQMGQiskitGenerator, backend: AerSimulator) -> None:
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


def _plot_eval(eval_rows: list[dict], out_dir: Path, warm_start_repair: int) -> None:
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

    vline = warm_start_repair if warm_start_repair > 0 else None

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
    _set_global_seeds(args.seed)
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
            print(f"[step {step}] phase={phase}")
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
    _set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.backend == "cudaq":
        if CudaQMGGenerator is None:
            raise RuntimeError("CUDA-Q backend requested but cudaq is not installed.")
        qmg = CudaQMGGenerator(
            atom_layers=args.atom_layers,
            bond_layers=args.bond_layers,
            repair_bonds=args.repair_bonds,
            device=_resolve_cudaq_device(args.device),
            seed=args.seed,
        )
        try:
            import cudaq  # noqa: WPS433

            print(
                f"[info] cudaq_version={getattr(cudaq, '__version__', 'unknown')} "
                f"target={getattr(qmg, 'target', 'unknown')} device={_resolve_cudaq_device(args.device)}"
            )
        except Exception:
            print("[warn] cudaq backend selected but unable to query cudaq details.")
    else:
        qmg = SQMGQiskitGenerator(
            atom_layers=args.atom_layers,
            bond_layers=args.bond_layers,
            repair_bonds=args.repair_bonds,
            seed=args.seed,
        )

    if args.backend == "qiskit":
        resolved_device, detected_gpus = _resolve_qiskit_device(args.device, 0)
        _print_aer_info(resolved_device)
        aer_backend, using_gpu = _create_aer_backend(args.seed, resolved_device, detected_gpus)
        _configure_generator_backend(qmg, aer_backend)
        if resolved_device == "gpu" and using_gpu:
            print(f"[info] Using Aer GPU backend (gpus={max(detected_gpus, 1)})")
        else:
            print("[info] Using Aer CPU backend")

    state_dim = build_state(qmg).size
    if args.backend == "cudaq":
        actor = CudaQQuantumActor(
            state_dim=state_dim,
            n_qubits=args.actor_qubits,
            n_layers=args.actor_layers,
            action_dim=args.action_dim,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            seed=args.seed,
        )
        critic = CudaQQuantumCritic(
            state_dim=state_dim,
            n_qubits=args.critic_qubits,
            n_layers=args.critic_layers,
            seed=args.seed + 1,
        )
    else:
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
        track_best=args.track_best,
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    eval_rows: list[dict] = []
    strict_eval_rewards: list[float] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_path = out_dir / "eval.csv"
        if not eval_path.exists():
            eval_path.write_text(
                "episode,phase,reward_pdf_eval,reward_raw_pdf_eval,validity_pdf_eval,uniqueness_pdf_eval,"
                "validity_raw_pdf_eval,uniqueness_raw_pdf_eval,sigma_max,k_batches,patience,eval_seed\n"
            )
        best_json_path = out_dir / "best_eval.json"

    prev_phase = None
    for step in range(1, args.steps + 1):
        phase = "repair" if step <= args.warm_start_repair else "strict"
        qmg.env.repair_bonds = phase == "repair"
        if args.track_best and prev_phase == "repair" and phase == "strict":
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
                f"reward_step={result['reward_step']:.6f} reward_main={result['reward_main']:.6f} "
                f"repeat_penalty={result['repeat_penalty']:.6f}"
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
                f"target_metric_pdf={stats_pdf['target_metric_pdf']:.6f} "
                f"reward_pdf={stats_pdf['reward_pdf']:.6f}"
            )
            print(
                f"raw_pdf: validity_raw={stats_pdf['validity_raw_pdf']:.4f} "
                f"uniqueness_raw={stats_pdf['uniqueness_raw_pdf']:.4f} "
                f"reward_raw={stats_pdf['reward_raw_pdf']:.6f}"
            )
            uniques = sorted(qmg.env.seen_smiles)[:10]
            print("Top unique SMILES (up to 10):")
            for smi in uniques:
                print(smi)
            print("-" * 60)

        if args.eval_every > 0 and step % args.eval_every == 0:
            weights_snapshot = qmg.get_weights()
            env_snap = _snapshot_env(qmg.env)
            eval_batch = qmg.sample_actions(batch_size=args.eval_shots)
            eval_stats = _eval_batch(eval_batch, repair_bonds=(phase == "repair"))
            _restore_env(qmg.env, env_snap)

            reward_pdf_eval = eval_stats["reward_pdf"]
            reward_raw_pdf_eval = eval_stats["reward_raw_pdf"]
            validity_pdf_eval = eval_stats["validity_pdf"]
            uniqueness_pdf_eval = eval_stats["uniqueness_pdf"]
            validity_raw_pdf_eval = eval_stats["validity_raw_pdf"]
            uniqueness_raw_pdf_eval = eval_stats["uniqueness_raw_pdf"]
            print(
                f"[eval step {step}] reward_pdf_eval={reward_pdf_eval:.6f} "
                f"reward_raw_pdf_eval={reward_raw_pdf_eval:.6f} "
                f"validity_pdf_eval={validity_pdf_eval:.4f} "
                f"uniqueness_pdf_eval={uniqueness_pdf_eval:.4f} "
                f"phase={phase}"
            )

            if args.adaptive_exploration and phase == "strict":
                strict_eval_rewards.append(float(reward_raw_pdf_eval))
                if len(strict_eval_rewards) > args.adapt_window:
                    strict_eval_rewards = strict_eval_rewards[-args.adapt_window :]
                if len(strict_eval_rewards) == args.adapt_window and all(
                    r < args.adapt_threshold for r in strict_eval_rewards
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

            if out_dir is not None:
                eval_row = {
                    "episode": step,
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
                    "eval_seed": args.seed,
                }
                eval_rows.append(eval_row)
                with (out_dir / "eval.csv").open("a", newline="") as handle:
                    handle.write(
                        f"{step},{phase},{reward_pdf_eval:.6f},{reward_raw_pdf_eval:.6f},"
                        f"{validity_pdf_eval:.6f},{uniqueness_pdf_eval:.6f},"
                        f"{validity_raw_pdf_eval:.6f},{uniqueness_raw_pdf_eval:.6f},"
                        f"{cfg.sigma_max:.6f},{cfg.k_batches},{cfg.patience},{args.seed}\n"
                    )

                metric_for_best = reward_raw_pdf_eval if phase == "strict" else reward_pdf_eval
                if cfg.best_weights is None or metric_for_best > cfg.best_reward_pdf:
                    cfg.best_reward_pdf = metric_for_best
                    cfg.best_weights = weights_snapshot.copy()
                    best_json = {
                        "episode": step,
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
                    np.save(out_dir / "best_weights.npy", cfg.best_weights)

            qmg.set_weights(weights_snapshot)
            print("-" * 60)

        prev_phase = phase

    if args.track_best and cfg.best_weights is not None and out_dir is None:
        out_path = Path("best_weights.npy")
        np.save(out_path, cfg.best_weights)
        print(f"Saved best weights to {out_path}")

    if out_dir is not None:
        _plot_eval(eval_rows, out_dir, args.warm_start_repair)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["helper", "a2c"], default="helper")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--backend", choices=["qiskit", "cudaq"], default="qiskit")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "cuda-cpu", "cuda-gpu"],
        default="auto",
    )
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
    parser.add_argument("--k-batches", type=int, default=2)
    parser.add_argument("--beta-novelty", type=float, default=0.0)
    parser.add_argument("--lambda-repeat", type=float, default=0.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-floor", type=float, default=0.0)
    parser.add_argument("--reward-clip-low", type=float, default=0.0)
    parser.add_argument("--reward-clip-high", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-boost", type=float, default=1.5)
    parser.add_argument("--sigma-decay", type=float, default=0.997)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    parser.add_argument("--actor-qubits", type=int, default=8)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-qubits", type=int, default=8)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--atom-layers", type=int, default=2)
    parser.add_argument("--bond-layers", type=int, default=1)
    parser.add_argument("--repair-bonds", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--out-dir", type=str, default="runs/a2c")

    args = parser.parse_args()

    if args.algo == "a2c":
        run_a2c(args)
    else:
        run_helper(args)


if __name__ == "__main__":
    main()
