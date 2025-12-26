from __future__ import annotations

from pathlib import Path

import pytest


def test_cudaq_shots_count_smoke(tmp_path: Path):
    pytest.importorskip("cudaq")

    from qmg.cudaq_generator import CudaQMGGenerator
    from scripts.run_one_train import run_one_train

    gen = CudaQMGGenerator(atom_layers=1, bond_layers=1, device="cpu", seed=123)
    assert gen.target in {"qpp-cpu", "nvidia"}
    batch = gen.sample_actions(batch_size=2)
    assert batch.atoms.shape == (2, 5)
    assert batch.bonds.shape[0] == 2

    out_dir = tmp_path / "cudaq_run"
    run_one_train(
        episodes=2,
        batch_size=2,
        out_dir=out_dir,
        backend="cudaq",
        device="cpu",
        atom_layers=1,
        bond_layers=1,
        actor_qubits=2,
        actor_layers=1,
        critic_qubits=2,
        critic_layers=1,
        action_dim=4,
        k_batches=1,
        eval_every=1,
        eval_shots=2,
        warm_start_repair=1,
        adaptive_exploration=False,
        log_every=1,
        seed=7,
    )
    eval_path = out_dir / "eval.csv"
    assert eval_path.exists()
    lines = eval_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    header = lines[0].split(",")
    assert "phase" in header
