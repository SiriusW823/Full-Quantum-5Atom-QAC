from __future__ import annotations

import pytest


def test_cudaq_shots_count_smoke():
    pytest.importorskip("cudaq")

    from qmg.cudaq_generator import CudaQMGGenerator

    gen = CudaQMGGenerator(atom_layers=1, bond_layers=1, device="cpu", seed=123)
    assert gen.target in {"qpp-cpu", "nvidia"}
    batch = gen.sample_actions(batch_size=2)
    assert batch.atoms.shape == (2, 5)
    assert batch.bonds.shape[0] == 2
