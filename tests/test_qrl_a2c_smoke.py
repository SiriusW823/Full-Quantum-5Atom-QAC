import numpy as np

from qmg.sqmg_generator import SQMGQiskitGenerator
from qrl.a2c import A2CConfig, a2c_step, build_state
from qrl.actor import QiskitQuantumActor
from qrl.critic import QiskitQuantumCritic


def test_qrl_a2c_smoke():
    rng = np.random.default_rng(123)
    gen = SQMGQiskitGenerator(atom_layers=1, bond_layers=1, seed=123)

    state = build_state(gen)
    actor = QiskitQuantumActor(state_dim=state.size, n_qubits=4, n_layers=1, action_dim=16, seed=123)
    critic = QiskitQuantumCritic(state_dim=state.size, n_qubits=4, n_layers=1, seed=456)

    proj = rng.normal(0.0, 1.0, size=(gen.num_weights, 16))
    proj /= np.sqrt(16)

    cfg = A2CConfig(
        action_dim=16,
        sigma=0.2,
        lr_theta=0.01,
        actor_a=0.02,
        actor_c=0.01,
        critic_a=0.02,
        critic_c=0.01,
    )

    result = a2c_step(
        gen=gen,
        actor=actor,
        critic=critic,
        proj=proj,
        rng=rng,
        batch_size=4,
        cfg=cfg,
        state=state,
    )

    assert "reward" in result
    assert "valid_ratio" in result
    assert "unique_ratio" in result
    assert "target_metric" in result
    assert actor.get_weights().shape == (actor.num_weights,)
    assert critic.get_weights().shape == (critic.num_weights,)
