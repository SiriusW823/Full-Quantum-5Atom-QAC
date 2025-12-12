import os
from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib.qrdqn import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from src.environment import MoleculeGenEnv


class RewardLogCallback(BaseCallback):
    """
    Logs episode rewards from infos['episode']['r'] during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards: List[float] = []
        self.timesteps: List[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)
        return True


def plot_convergence(timesteps: List[int], rewards: List[float], out_path: str = "training_convergence.png"):
    if not rewards:
        return
    window = max(1, len(rewards) // 50)
    ma = []
    dq = deque(maxlen=window)
    for r in rewards:
        dq.append(r)
        ma.append(sum(dq) / len(dq))

    plt.figure(figsize=(9, 4.5))
    plt.plot(timesteps, rewards, label="Episode reward", color="steelblue", alpha=0.6)
    plt.plot(timesteps, ma, label=f"Moving avg (w={window})", color="darkorange", linewidth=2.0)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("QRDQN Training Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_policy(model, env, n_episodes: int = 2000):
    total_valid = 0
    unique_valid = 0
    total_rewards = []
    seen: set[str] = set()
    all_smiles: List[str] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        if info.get("smiles"):
            all_smiles.append(info["smiles"])
        if info.get("valid", 0.0) >= 1.0:
            total_valid += 1
            if info.get("smiles") and info["smiles"] not in seen:
                unique_valid += 1
                seen.add(info["smiles"])
    validity_fraction = total_valid / n_episodes if n_episodes > 0 else 0.0
    uniqueness_fraction = (unique_valid / total_valid) if total_valid > 0 else 0.0
    golden_metric = validity_fraction * uniqueness_fraction
    return {
        "episodes": n_episodes,
        "avg_reward": float(np.mean(total_rewards)),
        "valid": total_valid,
        "unique": unique_valid,
        "validity_fraction": validity_fraction,
        "uniqueness_fraction": uniqueness_fraction,
        "golden_metric": golden_metric,
        "smiles": all_smiles,
    }


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    vec_env = make_vec_env(MoleculeGenEnv, n_envs=1)

    reward_cb = RewardLogCallback()

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        gamma=0.99,
        verbose=1,
    )
    model.learn(total_timesteps=100000, callback=reward_cb)

    # Final evaluation
    eval_env = MoleculeGenEnv()
    summary = evaluate_policy(model, eval_env, n_episodes=2000)

    # Plot convergence
    plot_convergence(reward_cb.timesteps, reward_cb.rewards, out_path="training_convergence.png")

    model.save("qrdqn_molecule_gen")

    # Final Summary output
    print("\n===== Final summary =====")
    print(f"Episodes: {summary['episodes']}")
    print(f"Average reward: {summary['avg_reward']:.4f}")
    print(f"Valid molecules: {summary['valid']}")
    print(f"Unique valid molecules: {summary['unique']}")
    print(f"Validity fraction: {summary['validity_fraction']:.4f}")
    print(f"Uniqueness fraction (unique/valid): {summary['uniqueness_fraction']:.4f}")
    print(f"Golden metric (Validity * Uniqueness): {summary['golden_metric']:.4f}")
    print("Saved convergence plot to training_convergence.png")


if __name__ == "__main__":
    main()
