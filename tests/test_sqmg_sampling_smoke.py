import numpy as np

from env import ATOM_VOCAB, BOND_VOCAB
from qmg.sqmg_generator import SQMGQiskitGenerator


def test_sqmg_sampling_smoke():
    gen = SQMGQiskitGenerator(atom_layers=1, bond_layers=1, shots=32, seed=123)
    batch = gen.sample_actions(batch_size=10)

    assert batch.atoms.shape == (10, 5)
    assert batch.bonds.shape == (10, 4)
    assert len(batch.smiles) == 10
    assert len(batch.valids) == 10
    assert len(batch.uniques) == 10

    assert np.all(batch.atoms >= 0) and np.all(batch.atoms < len(ATOM_VOCAB))
    assert np.all(batch.bonds >= 0) and np.all(batch.bonds < len(BOND_VOCAB))

    assert gen.env.samples == 10
