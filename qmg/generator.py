from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import Sampler

from env import ATOM_VOCAB, BOND_VOCAB, FiveAtomMolEnv
from qmg.circuit import build_pqc, encode_inputs


@dataclass
class SampledBatch:
    atoms: np.ndarray  # shape (batch, 5)
    bonds: np.ndarray  # shape (batch, 4)
    smiles: List[Optional[str]]
    valids: List[bool]


class QiskitQMGGenerator:
    """
    Factorized-head PQC sampler for 5-atom chains.
    Reuses a small PQC for each atom/bond head with different classical encodings.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        sampler: Sampler | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)
        self.env = FiveAtomMolEnv()

        qc, data_params, weight_params = build_pqc(n_qubits=n_qubits, n_layers=n_layers)
        self.base_circuit = qc
        self.data_params = data_params
        self.weight_params = weight_params

        # Trainable weights (can be optimized later)
        self.weights = self.rng.normal(0.0, 0.2, size=len(weight_params))

        self.sampler = sampler or Sampler()
        # Pre-transpile once for efficiency
        self._compiled = transpile(self.base_circuit, optimization_level=1)

    def _probs_for_head(self, head_id: int, num_categories: int) -> np.ndarray:
        noise = self.rng.normal(0.0, 1.0, size=self.n_qubits - 1)
        data_vals = encode_inputs(head_id=head_id, noise=noise, n_qubits=self.n_qubits)
        bind_dict = {self.data_params[i]: data_vals[i] for i in range(self.n_qubits)}
        for i, w in enumerate(self.weights):
            bind_dict[self.weight_params[i]] = w

        bound = self._compiled.assign_parameters(bind_dict, inplace=False)
        quasi = self.sampler.run(bound).result().quasi_dists[0]

        probs = np.zeros(num_categories, dtype=float)
        for idx in range(num_categories):
            probs[idx] = quasi.get(idx, 0.0)
        eps = 1e-6
        probs = probs + eps
        probs /= probs.sum()
        return probs

    def _sample_head(self, head_id: int, num_categories: int) -> Tuple[int, float]:
        probs = self._probs_for_head(head_id, num_categories)
        choice = int(self.rng.choice(num_categories, p=probs))
        logp = math.log(probs[choice])
        return choice, logp

    def sample_one(self) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        atoms = []
        bonds = []
        logps: List[float] = []
        # Atom heads: ids 0..4
        for i in range(5):
            a, lp = self._sample_head(head_id=i, num_categories=len(ATOM_VOCAB))
            atoms.append(a)
            logps.append(lp)
        # Bond heads: ids 100..103 to keep separation
        for i in range(4):
            b, lp = self._sample_head(head_id=100 + i, num_categories=len(BOND_VOCAB))
            bonds.append(b)
            logps.append(lp)
        return np.array(atoms, dtype=int), np.array(bonds, dtype=int), logps

    def sample_actions(self, batch_size: int = 1) -> SampledBatch:
        atoms_batch = np.zeros((batch_size, 5), dtype=int)
        bonds_batch = np.zeros((batch_size, 4), dtype=int)
        smiles: List[Optional[str]] = []
        valids: List[bool] = []
        for i in range(batch_size):
            atoms, bonds, _ = self.sample_one()
            atoms_batch[i] = atoms
            bonds_batch[i] = bonds
            s, v = self.env.build_smiles_from_actions(atoms.tolist(), bonds.tolist())
            smiles.append(s)
            valids.append(v)
        return SampledBatch(atoms=atoms_batch, bonds=bonds_batch, smiles=smiles, valids=valids)

    def to_smiles(self, atoms: Sequence[int], bonds: Sequence[int]) -> Tuple[Optional[str], bool]:
        return self.env.build_smiles_from_actions(list(atoms), list(bonds))
