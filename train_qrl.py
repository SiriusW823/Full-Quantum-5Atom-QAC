import os
import torch
import matplotlib.pyplot as plt
from typing import List

from src.environment import MoleculeGenEnv
from src.quantum_agent import QuantumPolicy


def moving_average(x: List[float], window: int = 20) -> List[float]:
    out = []
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out.append(sum(x[start : i + 1]) / (i - start + 1))
    return out


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env = MoleculeGenEnv()
    agent = QuantumPolicy(lr=1e-3)

    episodes = 2000
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        params_t, value_out = agent.sample_action()
        params_np = params_t.detach().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(params_np)

        agent.update(params_t, value_out, reward)

        rewards.append(reward)
        if (ep + 1) % 100 == 0:
            print(f"[ep {ep+1}] reward={reward:.3f} smiles={info.get('smiles')} unique={info.get('unique')}")

    ma = moving_average(rewards, window=50)
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(len(rewards)), rewards, label="Reward", alpha=0.6)
    plt.plot(range(len(ma)), ma, label="Moving avg (50)", color="darkorange")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Quantum Policy REINFORCE (experimental)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_convergence.png", dpi=200)
    plt.close()

    print("Training complete. Final unique count:", len(env.seen))


if __name__ == "__main__":
    main()
