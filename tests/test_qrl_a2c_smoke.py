import numpy as np

from qmg.sqmg_generator import SQMGQiskitGenerator
from qrl.a2c import A2CConfig, a2c_step, build_state
from qrl.actor import QiskitQuantumActor
from qrl.critic import QiskitQuantumCritic


def test_qrl_a2c_smoke():
    rng = np.random.default_rng(123)
    gen = SQMGQiskitGenerator(atom_layers=1, bond_layers=1, seed=123)

    state = build_state(gen)
    actor = QiskitQuantumActor(
        state_dim=state.size, n_qubits=4, n_layers=1, action_dim=16, sigma_min=0.05, sigma_max=0.5, seed=123
    )
    critic = QiskitQuantumCritic(state_dim=state.size, n_qubits=4, n_layers=1, seed=456)

    proj = rng.normal(0.0, 1.0, size=(gen.num_weights, 16))
    proj /= np.sqrt(16)

    cfg = A2CConfig(
        action_dim=16,
        lr_theta=0.01,
        actor_a=0.02,
        actor_c=0.01,
        critic_a=0.02,
        critic_c=0.01,
        k_batches=2,
        beta_novelty=0.05,
        lambda_repeat=0.10,
        ent_coef=0.01,
        reward_floor=-0.01,
        reward_clip_low=-0.05,
        reward_clip_high=1.0,
        sigma_min=0.05,
        sigma_max=0.5,
        sigma_boost=1.5,
        sigma_decay=0.995,
        patience=2,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
    )

    batch_size = 4
    result = a2c_step(
        gen=gen,
        actor=actor,
        critic=critic,
        proj=proj,
        rng=rng,
        batch_size=batch_size,
        cfg=cfg,
        state=state,
    )

    assert "reward" in result
    assert "validity_step" in result
    assert "uniqueness_step" in result
    assert "score_pdf_step" in result
    assert "novelty_step" in result
    assert "repeat_step" in result
    assert "reward_main" in result
    assert "repeat_penalty" in result
    assert "sigma" in result
    assert "entropy" in result
    assert "reward_avg" in result
    assert "unique_valid_in_batch" in result
    assert "novel_valid_in_batch" in result

    assert "ds" in result and result["ds"] == float(batch_size)
    assert "dv" in result and 0 <= result["dv"] <= result["ds"]
    assert int(result["dv"]) == result["dv"]
    assert int(result["unique_valid_in_batch"]) == result["unique_valid_in_batch"]
    assert int(result["novel_valid_in_batch"]) == result["novel_valid_in_batch"]

    validity_calc = result["dv"] / result["ds"] if result["ds"] else 0.0
    uniqueness_calc = result["unique_valid_in_batch"] / max(result["dv"], 1.0) if result["ds"] else 0.0
    score_calc = validity_calc * uniqueness_calc
    assert abs(result["validity_step"] - validity_calc) <= 1e-12
    assert abs(result["uniqueness_step"] - uniqueness_calc) <= 1e-12
    assert abs(result["score_pdf_step"] - score_calc) <= 1e-12
    assert 0.0 <= result["validity_step"] <= 1.0
    assert 0.0 <= result["uniqueness_step"] <= 1.0
    assert 0.0 <= result["score_pdf_step"] <= 1.0
    assert cfg.reward_clip_low <= result["reward"] <= cfg.reward_clip_high
    assert actor.get_weights().shape == (actor.num_weights,)
    assert critic.get_weights().shape == (critic.num_weights,)
