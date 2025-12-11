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
    entropies: List[torch.Tensor]
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

    def __init__(self, n_wires: int = 9, layers: int = 4, lr: float = 0.01, entropy_coef: float = 0.02):
        super().__init__()
        self.n_wires = n_wires
        self.entropy_coef = entropy_coef

        act_qnode, act_shapes = actor_qnode(n_wires=n_wires, layers=layers)
        crt_qnode, crt_shapes = critic_qnode(n_wires=n_wires, layers=layers)

        self.actor = qml.qnn.TorchLayer(act_qnode, weight_shapes=act_shapes)
        self.critic = qml.qnn.TorchLayer(crt_qnode, weight_shapes=crt_shapes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def policy(self, state_vec: torch.Tensor) -> D.Categorical:
        logits = self.actor(state_vec)
        return D.Categorical(logits=logits)

    def value(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.critic(state_vec)

    def act(self, history: List[int], epsilon: float = 0.1) -> Dict:
        state_vec = encode_state(history, n_wires=self.n_wires)
        dist = self.policy(state_vec)
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
        entropy = dist.entropy()
        value = self.value(state_vec)
        return {
            "action": int(action.item()),
            "logp": logp,
            "entropy": entropy,
            "value": value.squeeze(-1),
        }

    def rollout_episode(self, env: MoleculeEnv, seen: Set[str], epsilon: float = 0.1) -> EpisodeTrajectory:
        env.reset()
        history: List[int] = []
        logps: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        while True:
            step_out = self.act(history, epsilon=epsilon)
            env.step(step_out["action"])
            history = env.history

            logps.append(step_out["logp"])
            values.append(step_out["value"])
            entropies.append(step_out["entropy"])

            if env.done:
                break

        smiles, valid_flag, _ = env.finalize()
        unique_flag = 1.0 if valid_flag and smiles and smiles not in seen else 0.0
        if unique_flag:
            seen.add(smiles)

        return EpisodeTrajectory(
            logps=logps,
            values=values,
            entropies=entropies,
            smiles=smiles,
            valid=float(valid_flag),
            unique=float(unique_flag),
            length=len(history),
        )

    def update_batch(self, trajectories: List[EpisodeTrajectory], batch_reward: float, gamma: float = 0.99) -> Dict[str, float]:
        if not trajectories:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        actor_losses = []
        critic_losses = []
        entropy_vals = []

        for traj in trajectories:
            T = len(traj.logps)
            if T == 0:
                continue
            # Batch-level golden metric scaled by episode validity/uniqueness
            reward = batch_reward * (0.5 * traj.valid + 0.5 * traj.unique)
            # Single terminal reward propagated backwards
            returns = torch.tensor(
                [reward * (gamma ** (T - 1 - t)) for t in range(T)],
                dtype=torch.float32,
            )

            logps = torch.stack(traj.logps)
            values = torch.stack(traj.values).view(-1)
            ents = torch.stack(traj.entropies).view(-1)

            advantages = returns - values.detach()
            actor_loss = -(advantages * logps).mean() - self.entropy_coef * ents.mean()
            critic_loss = 0.5 * (returns - values).pow(2).mean()

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_vals.append(ents.mean().detach())

        if not actor_losses:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        actor_loss_tensor = torch.stack(actor_losses).mean()
        critic_loss_tensor = torch.stack(critic_losses).mean()
        loss = actor_loss_tensor + critic_loss_tensor

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss_tensor.item()),
            "critic_loss": float(critic_loss_tensor.item()),
            "entropy": float(torch.stack(entropy_vals).mean().item()),
        }
