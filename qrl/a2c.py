from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class A2CConfig:
    action_dim: int = 16
    sigma: float = 0.2
    lr_theta: float = 0.03
    actor_a: float = 0.05
    actor_c: float = 0.01
    critic_a: float = 0.05
    critic_c: float = 0.01


def _safe_stats(env) -> Dict[str, float]:
    if env is None:
        return {
            "samples": 0,
            "valid_count": 0,
            "unique_valid_count": 0,
            "valid_ratio": 0.0,
            "unique_ratio": 0.0,
            "target_metric": 0.0,
        }
    if hasattr(env, "stats"):
        return env.stats()
    samples = int(getattr(env, "samples", 0) or 0)
    valid = int(getattr(env, "valid_count", 0) or 0)
    unique = int(getattr(env, "unique_valid_count", 0) or 0)
    s = samples if samples > 0 else 1
    valid_ratio = valid / s
    unique_ratio = unique / s
    return {
        "samples": samples,
        "valid_count": valid,
        "unique_valid_count": unique,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "target_metric": valid_ratio * unique_ratio,
    }


def build_state(gen, stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    env = getattr(gen, "env", None)
    stats = stats or _safe_stats(env)

    samples = int(stats.get("samples", 0) or 0)
    valid = int(stats.get("valid_count", 0) or 0)
    unique = int(stats.get("unique_valid_count", 0) or 0)
    s = samples if samples > 0 else 1

    valid_ratio = float(stats.get("valid_ratio", valid / s))
    unique_ratio = float(stats.get("unique_ratio", unique / s))
    target_metric = float(stats.get("target_metric", valid_ratio * unique_ratio))

    dup_ratio = max(valid - unique, 0) / s
    sample_scale = np.log1p(samples) / 10.0

    theta = gen.get_weights()
    if theta.size:
        theta_mean = float(theta.mean())
        theta_std = float(theta.std())
        theta_l2 = float(np.linalg.norm(theta) / np.sqrt(theta.size))
        theta_min = float(theta.min())
        theta_max = float(theta.max())
    else:
        theta_mean = theta_std = theta_l2 = theta_min = theta_max = 0.0

    state = np.array(
        [
            valid_ratio,
            unique_ratio,
            target_metric,
            valid / s,
            unique / s,
            dup_ratio,
            sample_scale,
            theta_mean,
            theta_std,
            theta_l2,
            theta_min,
            theta_max,
        ],
        dtype=float,
    )
    return np.tanh(state)


def gaussian_logprob(action: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    action = np.asarray(action, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if action.shape != mu.shape:
        raise ValueError("action and mu must have the same shape")
    sigma = float(max(sigma, 1e-6))
    diff = (action - mu) / sigma
    return float(-0.5 * np.sum(diff**2 + np.log(2.0 * np.pi * sigma**2)))


def spsa_step(
    params: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    a: float,
    c: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    delta = rng.choice([-1.0, 1.0], size=params.shape)
    loss_plus = float(loss_fn(params + c * delta))
    loss_minus = float(loss_fn(params - c * delta))
    ghat = (loss_plus - loss_minus) / (2.0 * c) * delta
    new_params = params - a * ghat
    loss_est = 0.5 * (loss_plus + loss_minus)
    return new_params, loss_est


def _eval_with_weights(model, weights: np.ndarray, state: np.ndarray):
    base = model.get_weights()
    model.set_weights(weights)
    out = model.forward(state)
    model.set_weights(base)
    return out


def a2c_step(
    gen,
    actor,
    critic,
    proj: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    cfg: A2CConfig,
    state: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if proj.shape[1] != cfg.action_dim:
        raise ValueError("projection matrix action_dim mismatch")

    state = state if state is not None else build_state(gen)
    mu = actor.forward(state)
    if mu.shape[0] != cfg.action_dim:
        raise ValueError("actor output dimension mismatch")

    sigma = float(max(cfg.sigma, 1e-6))
    action = rng.normal(mu, sigma)
    action = np.clip(action, -1.0, 1.0)

    delta = proj @ action
    norm = float(np.linalg.norm(delta))
    if norm > 0:
        delta = delta / norm
    theta = gen.get_weights()
    gen.set_weights(theta + cfg.lr_theta * delta)

    before = _safe_stats(gen.env)
    gen.sample_actions(batch_size=batch_size)
    after = _safe_stats(gen.env)

    ds = int(after["samples"] - before["samples"])
    dv = int(after["valid_count"] - before["valid_count"])
    du = int(after["unique_valid_count"] - before["unique_valid_count"])
    if ds <= 0:
        ds = 1
    if dv < 0:
        dv = 0
    if du < 0:
        du = 0

    valid_ratio_step = dv / ds
    unique_ratio_step = du / ds
    target_metric_step = valid_ratio_step * unique_ratio_step
    reward = float(target_metric_step)

    value = float(critic.forward(state))
    advantage = reward - value

    def critic_loss_fn(w: np.ndarray) -> float:
        v = float(_eval_with_weights(critic, w, state))
        return (v - reward) ** 2

    new_cw, critic_loss = spsa_step(
        critic.get_weights(), critic_loss_fn, cfg.critic_a, cfg.critic_c, rng
    )
    critic.set_weights(new_cw)

    def actor_loss_fn(w: np.ndarray) -> float:
        mu_w = _eval_with_weights(actor, w, state)
        logp = gaussian_logprob(action, mu_w, sigma)
        return -advantage * logp

    new_aw, actor_loss = spsa_step(
        actor.get_weights(), actor_loss_fn, cfg.actor_a, cfg.actor_c, rng
    )
    actor.set_weights(new_aw)

    return {
        "reward": reward,
        "valid_ratio": float(valid_ratio_step),
        "unique_ratio": float(unique_ratio_step),
        "target_metric": float(target_metric_step),
        "value": float(value),
        "advantage": float(advantage),
        "actor_loss": float(actor_loss),
        "critic_loss": float(critic_loss),
    }


__all__ = ["A2CConfig", "build_state", "gaussian_logprob", "spsa_step", "a2c_step"]
