import numpy as np

from qrl.helper import QiskitQRLHelper


def test_qrl_helper_smoke():
    helper = QiskitQRLHelper(seed=123)
    smiles_list = ["CCCCC", "CCNCC", "COCOC"]
    scores = helper.score(smiles_list)
    assert scores.shape == (3,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    targets = np.array([1.0, 0.0, 1.0])
    result = helper.train_step(smiles_list, targets, lr=0.05, spsa_eps=0.02)
    assert result.loss >= 0.0
    assert result.scores.shape == (3,)
    assert np.all(result.scores >= 0.0) and np.all(result.scores <= 1.0)
