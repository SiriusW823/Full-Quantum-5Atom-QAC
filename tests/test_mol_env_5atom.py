import math

from env import ATOM_VOCAB, BOND_VOCAB, EDGE_LIST, FiveAtomMolEnv


def test_env_valid_known_action():
    env = FiveAtomMolEnv()
    atoms = [ATOM_VOCAB.index("C")] * 5
    bonds = [BOND_VOCAB.index("SINGLE")] * len(EDGE_LIST)

    smiles, valid = env.build_smiles_from_actions(atoms, bonds)

    assert valid is True
    assert smiles is not None
    assert env.metrics.samples == 1
    assert env.metrics.valid_count == 1
    assert env.metrics.unique_valid_count == 1
    assert math.isclose(env.target_metric, 1.0, rel_tol=1e-6)


def test_env_invalid_when_active_lt_2():
    env = FiveAtomMolEnv()

    # all NONE -> active atoms < 2 -> invalid
    atoms = [ATOM_VOCAB.index("NONE")] * 5
    bonds = [BOND_VOCAB.index("SINGLE")] * len(EDGE_LIST)
    smiles, valid = env.build_smiles_from_actions(atoms, bonds)
    assert valid is False
    assert smiles is None

    # only one active atom -> invalid
    atoms = [ATOM_VOCAB.index("C")] + [ATOM_VOCAB.index("NONE")] * 4
    smiles, valid = env.build_smiles_from_actions(atoms, bonds)
    assert valid is False
    assert smiles is None


def test_env_rejects_fragment():
    env = FiveAtomMolEnv(enforce_single_fragment=True)
    atoms = [ATOM_VOCAB.index("C")] * 5
    # no bonds -> multiple fragments -> rejected
    bonds = [BOND_VOCAB.index("NONE")] * len(EDGE_LIST)
    smiles, valid = env.build_smiles_from_actions(atoms, bonds)
    assert valid is False
    assert smiles is None


def test_env_uniqueness_updates_only_when_valid():
    env = FiveAtomMolEnv()
    atoms = [ATOM_VOCAB.index("C")] * 5
    bonds = [BOND_VOCAB.index("SINGLE")] * len(EDGE_LIST)

    smiles1, valid1 = env.build_smiles_from_actions(atoms, bonds)
    assert valid1 and smiles1
    assert env.metrics.valid_count == 1
    assert env.metrics.unique_valid_count == 1

    smiles2, valid2 = env.build_smiles_from_actions(atoms, bonds)
    assert valid2 and smiles2
    assert env.metrics.valid_count == 2
    assert env.metrics.unique_valid_count == 1

    # invalid action shouldn't change unique count
    bonds_fragment = [BOND_VOCAB.index("NONE")] * len(EDGE_LIST)
    env.build_smiles_from_actions(atoms, bonds_fragment)
    assert env.metrics.unique_valid_count == 1
