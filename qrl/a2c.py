from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Set

import numpy as np

from qrl.actor import gaussian_entropy, gaussian_logprob


@dataclass
class A2CConfig:
    action_dim: int = 16
    lr_theta: float = 0.03
    actor_a: float = 0.05
    actor_c: float = 0.01
    critic_a: float = 0.05
    critic_c: float = 0.01
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    k_batches: int = 2
    beta_novelty: float = 0.0
    lambda_repeat: float = 0.0
    ent_coef: float = 0.01
    reward_floor: float = 0.0
    reward_clip_low: float = 0.0
    reward_clip_high: float = 1.0
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    sigma_boost: float = 1.25
    sigma_decay: float = 0.997
    patience: int = 50
    sigma_current: Optional[float] = None
    reward_ema_beta: float = 0.99
    reward_ema: float = 0.0
    step_idx: int = 0
    adv_count: int = 0
    adv_mean: float = 0.0
    adv_m2: float = 0.0
    no_unique_steps: int = 0
    seen_valid_smiles: Set[str] = field(default_factory=set)
    reward_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2))


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
    log_unique = np.log1p(unique) / max(1.0, np.log1p(samples))

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
            log_unique,
            theta_mean,
            theta_std,
            theta_l2,
            theta_min,
            theta_max,
        ],
        dtype=float,
    )
    return np.tanh(state)


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
    mu, sigma_actor = actor.forward(state)
    if mu.shape[0] != cfg.action_dim:
        raise ValueError("actor output dimension mismatch")

    sigma_base = float(np.clip(sigma_actor, cfg.sigma_min, cfg.sigma_max))
    if cfg.sigma_current is None:
        cfg.sigma_current = sigma_base
    sigma = float(np.clip(cfg.sigma_current, cfg.sigma_min, cfg.sigma_max))
    action = rng.normal(mu, sigma)
    action = np.clip(action, -1.0, 1.0)

    delta = proj @ action
    norm = float(np.linalg.norm(delta))
    if norm > 0:
        delta = delta / norm
    theta = gen.get_weights()
    gen.set_weights(theta + cfg.lr_theta * delta)

    reward_k_list = []
    ds = int(batch_size)
    dv_any = False
    last_counts = None

    for k_idx in range(max(1, cfg.k_batches)):
        batch = gen.sample_actions(batch_size=batch_size)
        valid_smiles = [s for s, v in zip(batch.smiles, batch.valids) if v and s]
        dv = len(valid_smiles)
        unique_valid_in_batch = len(set(valid_smiles))
        novel_valid_in_batch = sum(1 for s in valid_smiles if s not in cfg.seen_valid_smiles)
        if valid_smiles:
            cfg.seen_valid_smiles.update(valid_smiles)

        validity_step = dv / ds if ds else 0.0
        uniqueness_pdf_step = (unique_valid_in_batch / dv) if dv > 0 else 0.0
        score_pdf_step = validity_step * uniqueness_pdf_step
        reward_step = score_pdf_step
        novelty_step = (novel_valid_in_batch / dv) if dv > 0 else 0.0
        repeat_step = ((dv - unique_valid_in_batch) / ds) if ds else 0.0

        reward_k = reward_step + cfg.beta_novelty * novelty_step - cfg.lambda_repeat * repeat_step
        if dv == 0:
            reward_k = cfg.reward_floor
        reward_k = float(np.clip(reward_k, cfg.reward_clip_low, cfg.reward_clip_high))
        reward_k_list.append(reward_k)

        if dv > 0:
            dv_any = True

        if k_idx == max(1, cfg.k_batches) - 1:
            last_counts = (
                dv,
                unique_valid_in_batch,
                novel_valid_in_batch,
                validity_step,
                uniqueness_pdf_step,
                score_pdf_step,
                reward_step,
                novelty_step,
                repeat_step,
            )

    if last_counts is None:
        last_counts = (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    dv, unique_valid_in_batch, novel_valid_in_batch, validity_step, uniqueness_pdf_step, score_pdf_step, reward_step, novelty_step, repeat_step = last_counts

    reward_step_mean = float(np.mean(reward_k_list)) if reward_k_list else 0.0
    if cfg.reward_history.maxlen != max(1, cfg.k_batches):
        cfg.reward_history = deque(cfg.reward_history, maxlen=max(1, cfg.k_batches))
    cfg.reward_history.append(reward_step_mean)
    reward_avg = float(np.mean(cfg.reward_history)) if cfg.reward_history else reward_step_mean
    reward = reward_step_mean

    if unique_valid_in_batch == 0:
        cfg.no_unique_steps += 1
    else:
        cfg.no_unique_steps = 0

    if cfg.no_unique_steps >= cfg.patience:
        cfg.sigma_current = min(sigma * cfg.sigma_boost, cfg.sigma_max)
    else:
        cfg.sigma_current = max(sigma * cfg.sigma_decay, cfg.sigma_min)

    value = float(critic.forward(state))
    cfg.reward_ema = cfg.reward_ema_beta * cfg.reward_ema + (1.0 - cfg.reward_ema_beta) * reward_avg
    advantage = (reward_avg - cfg.reward_ema) - value + 0.5 * value
    cfg.adv_count += 1
    delta = advantage - cfg.adv_mean
    cfg.adv_mean += delta / cfg.adv_count
    delta2 = advantage - cfg.adv_mean
    cfg.adv_m2 += delta * delta2
    if cfg.adv_count > 1:
        adv_std = float(np.sqrt(cfg.adv_m2 / (cfg.adv_count - 1)))
    else:
        adv_std = 1.0
    adv_norm = (advantage - cfg.adv_mean) / (adv_std + 1e-8)
    adv_norm = float(np.clip(adv_norm, -3.0, 3.0))

    k = cfg.step_idx
    cfg.step_idx += 1
    actor_ak = cfg.actor_a / ((k + 1) ** cfg.spsa_alpha)
    actor_ck = cfg.actor_c / ((k + 1) ** cfg.spsa_gamma)
    critic_ak = cfg.critic_a / ((k + 1) ** cfg.spsa_alpha)
    critic_ck = cfg.critic_c / ((k + 1) ** cfg.spsa_gamma)

    def critic_loss_fn(w: np.ndarray) -> float:
        v = float(_eval_with_weights(critic, w, state))
        return (v - reward_avg) ** 2

    new_cw, critic_loss = spsa_step(
        critic.get_weights(), critic_loss_fn, critic_ak, critic_ck, rng
    )
    critic.set_weights(new_cw)

    actor_loss = 0.0
    entropy = gaussian_entropy(cfg.action_dim, sigma)
    if dv_any:
        def actor_loss_fn(w: np.ndarray) -> float:
            mu_w, sigma_w = _eval_with_weights(actor, w, state)
            sigma_w = float(np.clip(sigma_w, cfg.sigma_min, cfg.sigma_max))
            logp = gaussian_logprob(action, mu_w, sigma_w)
            ent = gaussian_entropy(cfg.action_dim, sigma_w)
            return -adv_norm * logp - cfg.ent_coef * ent

        new_aw, actor_loss = spsa_step(
            actor.get_weights(), actor_loss_fn, actor_ak, actor_ck, rng
        )
        actor.set_weights(new_aw)

    return {
        "reward": float(reward),
        "reward_avg": float(reward_avg),
        "reward_step": float(reward_step),
        "reward_main": float(score_pdf_step),
        "repeat_penalty": float(cfg.lambda_repeat * repeat_step),
        "validity_step": float(validity_step),
        "uniqueness_step": float(uniqueness_pdf_step),
        "score_pdf_step": float(score_pdf_step),
        "novelty_step": float(novelty_step),
        "repeat_step": float(repeat_step),
        "ds": int(ds),
        "dv": int(dv),
        "du": int(unique_valid_in_batch),
        "unique_valid_in_batch": int(unique_valid_in_batch),
        "novel_valid_in_batch": int(novel_valid_in_batch),
        "value": float(value),
        "advantage": float(advantage),
        "actor_loss": float(actor_loss),
        "critic_loss": float(critic_loss),
        "sigma": float(sigma),
        "entropy": float(entropy),
        "valid_ratio_step": float(validity_step),
        "unique_ratio_step": float(uniqueness_pdf_step),
        "composite_step": float(score_pdf_step),
        "repeat_ratio_step": float(repeat_step),
    }


__all__ = ["A2CConfig", "build_state", "spsa_step", "a2c_step"]
