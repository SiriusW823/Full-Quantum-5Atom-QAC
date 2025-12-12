import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import MoleculeGenEnv


def evaluate(model, env, n_episodes: int = 5):
    rewards = []
    smiles_samples = []
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
        rewards.append(ep_reward)
        if info.get("smiles"):
            smiles_samples.append(info["smiles"])
    avg_reward = np.mean(rewards) if rewards else 0.0
    print(f"Eval episodes: {n_episodes}, avg reward: {avg_reward:.4f}, samples: {smiles_samples[:3]}")


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    vec_env = make_vec_env(MoleculeGenEnv, n_envs=1)

    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, verbose=1)
    model.learn(total_timesteps=50000)

    # Evaluate on a fresh env (non-vectorized)
    eval_env = MoleculeGenEnv()
    evaluate(model, eval_env, n_episodes=5)

    model.save("ppo_molecule_gen")
    print("Training complete. Model saved to ppo_molecule_gen.")


if __name__ == "__main__":
    main()
