import numpy as np

from qmg.generator import QiskitQMGGenerator


def test_qmg_sampling_smoke():
    gen = QiskitQMGGenerator(seed=42)
    batch = gen.sample_actions(batch_size=10)

    assert batch.atoms.shape == (10, 5)
    assert batch.bonds.shape == (10, 4)

    # Ensure env path executes without raising and metrics track
    assert gen.env.metrics.samples == 10
    # valid_count may be zero but should be >= 0
    assert gen.env.metrics.valid_count >= 0

    # Verify no NaNs in probabilities by re-sampling one head
    atoms, bonds, _ = gen.sample_one()
    assert atoms.shape == (5,)
    assert bonds.shape == (4,)
    assert not np.isnan(atoms).any()
    assert not np.isnan(bonds).any()
