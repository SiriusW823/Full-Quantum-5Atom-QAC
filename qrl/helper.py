from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import Sampler

from qrl.circuit import build_qrl_pqc
from qrl.features import smiles_to_features


@dataclass
class QRLBatchResult:
    scores: np.ndarray
    loss: float


class QiskitQRLHelper:
    """
    Quantum helper (prior/critic) that scores SMILES in [0,1].
    Uses SPSA for training to fit novelty targets.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_dim: int = 12,
        sampler: Sampler | None = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler or Sampler()

        qc, data_params, weight_params = build_qrl_pqc(n_qubits, n_layers)
        self.base_circuit = qc
        self.data_params = data_params
        self.weight_params = weight_params
        self.weights = self.rng.normal(0.0, 0.2, size=len(weight_params))
        self._compiled = transpile(self.base_circuit, optimization_level=1)
        self._feat_cache: Dict[str, np.ndarray] = {}

    def _features(self, smiles: str) -> np.ndarray:
        if smiles in self._feat_cache:
            return self._feat_cache[smiles]
        feats = smiles_to_features(smiles, dim=self.feature_dim)
        self._feat_cache[smiles] = feats
        return feats

    def _encode(self, feats: np.ndarray) -> Dict:
        angles = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_qubits):
            idx = i % feats.shape[0]
            angles[i] = float(feats[idx])
        bind = {self.data_params[i]: angles[i] for i in range(self.n_qubits)}
        for i, w in enumerate(self.weights):
            bind[self.weight_params[i]] = w
        return bind

    def _expectation_from_quasi(self, quasi) -> float:
        # use Z expectation on qubit 0
        exp_z = 0.0
        norm = 0.0
        for bitstring, prob in quasi.items():
            if isinstance(bitstring, str):
                b = bitstring
            else:
                b = format(bitstring, f"0{self.n_qubits}b")
            sign = 1.0 if b[-1] == "0" else -1.0  # last char corresponds to qubit 0
            exp_z += sign * prob
            norm += prob
        if norm == 0:
            return 0.5
        exp_z /= norm
        val = (1.0 - exp_z) / 2.0
        return float(np.clip(val, 0.0, 1.0))

    def score(self, smiles_list: Iterable[str]) -> np.ndarray:
        scores: List[float] = []
        for smi in smiles_list:
            feats = self._features(smi) if smi else np.zeros(self.feature_dim, dtype=float)
            bind = self._encode(feats)
            bound = self._compiled.assign_parameters(bind)
            quasi = self.sampler.run(bound).result().quasi_dists[0]
            scores.append(self._expectation_from_quasi(quasi))
        return np.array(scores, dtype=float)

    def train_step(
        self, smiles_list: List[str], targets: np.ndarray, lr: float = 0.1, spsa_eps: float = 0.05
    ) -> QRLBatchResult:
        if len(smiles_list) == 0:
            return QRLBatchResult(scores=np.array([]), loss=0.0)

        base_scores = self.score(smiles_list)
        loss_base = float(np.mean((base_scores - targets) ** 2))

        delta = self.rng.choice([-1.0, 1.0], size=self.weights.shape)
        w_plus = self.weights + spsa_eps * delta
        w_minus = self.weights - spsa_eps * delta

        # plus
        self.weights = w_plus
        scores_plus = self.score(smiles_list)
        loss_plus = float(np.mean((scores_plus - targets) ** 2))

        # minus
        self.weights = w_minus
        scores_minus = self.score(smiles_list)
        loss_minus = float(np.mean((scores_minus - targets) ** 2))

        # gradient estimate
        ghat = (loss_plus - loss_minus) / (2.0 * spsa_eps) * delta
        self.weights = self.weights - lr * ghat

        # return to base weights after update
        scores_updated = self.score(smiles_list)
        loss_updated = float(np.mean((scores_updated - targets) ** 2))

        return QRLBatchResult(scores=scores_updated, loss=loss_updated)
