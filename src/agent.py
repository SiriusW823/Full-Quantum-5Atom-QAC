from dataclasses import dataclass
from typing import List, Set, Dict
import random

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

from src.circuits import actor_qnode, critic_qnode
from src.embedding import encode_state
from src.environment import MoleculeEnv


@dataclass
class EpisodeTrajectory:
    logps: List[torch.Tensor]
    values: List[torch.Tensor]
    policy_probs: List[torch.Tensor]
    reward: float
    smiles: str | None
    valid: float
    unique: float
    length: int


class QuantumActorCritic(nn.Module):
    """
    Full quantum actor-critic (two VQCs, no classical dense layers).
    - Actor: outputs 4 logits via expectation values (softmaxed by the categorical dist).
    - Critic: outputs scalar V(s) for advantage estimation.
    """

    def __init__(self, n_wires: int = 9, max_layers: int = 5, lr: float = 0.0001, entropy_beta: float = 0.01):
        super().__init__()
        self.n_wires = n_wires
        self.entropy_beta = entropy_beta
        self.max_layers = max_layers

        self.actor_qnode = actor_qnode(n_wires=n_wires)
        self.critic_qnode = critic_qnode(n_wires=n_wires)

        # Trainable quantum parameters (shared across dynamic depths)
        self.actor_weights = nn.Parameter(0.01 * torch.randn(max_layers, n_wires, 3))
        self.critic_weights = nn.Parameter(0.01 * torch.randn(max_layers, n_wires, 3))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _dynamic_layers(self, state_vec: torch.Tensor) -> int:
        """
        Compute dynamic depth based on active (non-zero) angles in the state vector.
        Scales active steps (0-9) to layers in [2, max_layers] (ceil).
        """
        active_steps = (state_vec != 0).sum().float()
        num_layers = torch.ceil(2 + (active_steps / 9.0) * 3.0)
        return int(torch.clamp(num_layers, 2, self.max_layers).item())

    def policy(self, state_vec: torch.Tensor):
        num_layers = self._dynamic_layers(state_vec)
        weights = self.actor_weights[:num_layers]
        logits = self.actor_qnode(state_vec, weights)
        if isinstance(logits, (list, tuple)):
            logits = torch.stack([torch.as_tensor(l) for l in logits])
        probs = torch.softmax(logits, dim=-1)
        dist = D.Categorical(probs=probs)
        return dist, logits, probs, num_layers

    def value(self, state_vec: torch.Tensor, num_layers: int) -> torch.Tensor:
        weights = self.critic_weights[:num_layers]
        return self.critic_qnode(state_vec, weights)

    def act(self, history: List[int], epsilon: float = 0.1) -> Dict:
        state_vec = encode_state(history, n_wires=self.n_wires)
        dist, logits, probs, num_layers = self.policy(state_vec)
        action = dist.sample()

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            if len(history) == 0:
                action = torch.tensor(random.choice([1, 2, 3]))
            else:
                action = torch.tensor(random.randint(0, 3))

        # Prevent trivial empty molecule
        if len(history) == 0 and action.item() == 0:
            action = torch.tensor(random.choice([1, 2, 3]))

        logp = dist.log_prob(action)
        value = self.value(state_vec, num_layers=num_layers)
        return {
            "action": int(action.item()),
            "logp": logp,
            "policy_probs": probs,
            "value": value.squeeze(-1),
            "num_layers": num_layers,
        }

    def rollout_episode(self, env: MoleculeEnv, seen: Set[str], epsilon: float = 0.1) -> EpisodeTrajectory:
        env.reset()
        history: List[int] = []
        logps: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        policy_probs: List[torch.Tensor] = []

        while True:
            step_out = self.act(history, epsilon=epsilon)
            env.step(step_out["action"])
            history = env.history

            logps.append(step_out["logp"])
            values.append(step_out["value"])
            policy_probs.append(step_out["policy_probs"])

            if env.done:
                break

        smiles, valid_flag, _ = env.finalize()
        unique_flag = 1.0 if valid_flag and smiles and smiles not in seen else 0.0
        if unique_flag:
            seen.add(smiles)
        reward = env.shaped_reward(float(valid_flag), unique_flag)

        return EpisodeTrajectory(
            logps=logps,
            values=values,
            policy_probs=policy_probs,
            reward=reward,
            smiles=smiles,
            valid=float(valid_flag),
            unique=float(unique_flag),
            length=len(history),
        )

    def update_batch(self, trajectories: List[EpisodeTrajectory], gamma: float = 0.99) -> Dict[str, float]:
        if not trajectories:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        actor_losses = []
        critic_losses = []
        entropy_vals = []

        for traj in trajectories:
            T = len(traj.logps)
            if T == 0:
                continue
            # Reward shaping per episode
            reward = traj.reward
            # Single terminal reward propagated backwards
            returns = torch.tensor(
                [reward * (gamma ** (T - 1 - t)) for t in range(T)],
                dtype=torch.float32,
            )

            logps = torch.stack(traj.logps)
            values = torch.stack(traj.values).view(-1)
            probs = torch.stack(traj.policy_probs)

            # Policy entropy using explicit softmax probabilities
            entropy_term = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()

            advantages = returns - values.detach()
            actor_loss = -(advantages * logps).mean() - self.entropy_beta * entropy_term
            critic_loss = 0.5 * (returns - values).pow(2).mean()

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_vals.append(entropy_term.detach())

        if not actor_losses:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        actor_loss_tensor = torch.stack(actor_losses).mean()
        critic_loss_tensor = torch.stack(critic_losses).mean()
        loss = actor_loss_tensor + critic_loss_tensor

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss_tensor.item()),
            "critic_loss": float(critic_loss_tensor.item()),
            "entropy": float(torch.stack(entropy_vals).mean().item()),
        }
