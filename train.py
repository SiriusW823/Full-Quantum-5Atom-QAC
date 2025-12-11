import os
import random
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import torch

from src.agent import QuantumActorCritic
from src.environment import MoleculeEnv


def set_seed(seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)


def moving_average(series: List[float], window: int = 10) -> List[float]:
    if window <= 1:
        return series
    smoothed = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        smoothed.append(sum(series[start : i + 1]) / (i - start + 1))
    return smoothed


def summarize_smiles(smiles_batch: List[str]) -> str:
    if not smiles_batch:
        return "None"
    counts = Counter(smiles_batch)
    top = counts.most_common(3)
    return "; ".join([f"{s} (x{c})" for s, c in top])


def plot_convergence(episodes_axis: List[int], scores: List[float], window: int = 10, out_path: str = "training_convergence.png"):
    if not scores:
        return
    plt.figure(figsize=(9, 4.5))
    plt.plot(episodes_axis, scores, label="Batch score", color="steelblue", alpha=0.75)
    plt.plot(episodes_axis, moving_average(scores, window), label=f"Moving avg (w={window})", color="darkorange", linewidth=2.0)
    plt.xlabel("Training Episodes")
    plt.ylabel("Score = (Valid/Samples) * (Unique/Samples)")
    plt.title("Quantum RL Convergence (Golden Metric)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train():
    set_seed(7)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Hyperparameters
    episodes = 2000
    batch_size = 32
    epsilon = 0.15
    ma_window = 10

    env = MoleculeEnv()
    agent = QuantumActorCritic(n_wires=9, layers=4, lr=0.01, entropy_coef=0.02)
    seen: set[str] = set()

    episode_marks: List[int] = []
    batch_scores: List[float] = []

    total_valid = 0
    total_unique = 0
    episodes_done = 0
    log_bucket = 0

    while episodes_done < episodes:
        current_batch = min(batch_size, episodes - episodes_done)
        trajectories = []
        batch_valid_flags: List[float] = []
        batch_unique_flags: List[float] = []
        smiles_batch: List[str] = []

        for _ in range(current_batch):
            traj = agent.rollout_episode(env, seen, epsilon=epsilon)
            trajectories.append(traj)
            batch_valid_flags.append(traj.valid)
            batch_unique_flags.append(traj.unique)
            if traj.valid and traj.smiles:
                smiles_batch.append(traj.smiles)

        total_valid += int(sum(batch_valid_flags))
        total_unique += int(sum(batch_unique_flags))

        batch_score = MoleculeEnv.golden_metric(batch_valid_flags, batch_unique_flags, current_batch)
        loss_stats = agent.update_batch(trajectories, batch_reward=batch_score, gamma=0.99)

        episodes_done += current_batch
        episode_marks.append(episodes_done)
        batch_scores.append(batch_score)

        if (episodes_done // 50) > log_bucket or episodes_done == episodes:
            log_bucket = episodes_done // 50
            print(
                f"[ep {episodes_done:04d}] batch_score={batch_score:.4f} "
                f"valid={int(sum(batch_valid_flags))}/{current_batch} unique_valid={int(sum(batch_unique_flags))}/{current_batch} "
                f"actor_loss={loss_stats['actor_loss']:.4f} critic_loss={loss_stats['critic_loss']:.4f} "
                f"top3={summarize_smiles(smiles_batch)}"
            )

    final_score = (total_valid / episodes) * (total_unique / episodes)
    print("\n===== Final summary =====")
    print(f"Episodes: {episodes}")
    print(f"Valid molecules: {total_valid}")
    print(f"Unique valid molecules: {total_unique}")
    print(f"Golden metric (Valid/Samples * Unique/Samples): {final_score:.4f}")

    plot_convergence(episode_marks, batch_scores, window=ma_window, out_path="training_convergence.png")
    print("Saved convergence plot to training_convergence.png")


if __name__ == "__main__":
    train()
