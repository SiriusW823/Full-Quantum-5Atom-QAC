import numpy as np

from qmg.sqmg_generator import SQMGQiskitGenerator
from env import ATOM_VOCAB


def test_sqmg_sampling_smoke():
    gen = SQMGQiskitGenerator(n_layers_atom=1, n_layers_bond=1, shots=64, seed=123)
    batch = gen.sample_actions(batch_size=10)

    assert batch.atoms.shape == (10, 5)
    assert batch.bonds.shape == (10, 4)
    assert len(batch.uniques) == 10

    atom_ids = batch.atoms.flatten()
    assert np.all(atom_ids >= 0)
    assert np.all(atom_ids < len(ATOM_VOCAB))

    assert gen.env.samples == 10
    assert gen.env.valid_count >= 0
