from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import Sampler

from env import ATOM_VOCAB, BOND_VOCAB, FiveAtomMolEnv
from qmg.generator import SampledBatch
from qmg.sqmg_circuit import ATOM_Q, BOND_Q, N_ATOMS, build_sqmg_circuit

ATOM_STATE_MAP = {0: "C", 1: "N", 2: "O"}


class SQMGQiskitGenerator:
    """
    SQMG-style generator using a single 23-qubit PQC (3 qubits/atom, 2 qubits/bond).
    """

    def __init__(
        self,
        n_layers_atom: int = 2,
        n_layers_bond: int = 2,
        sampler: Sampler | None = None,
        seed: int | None = None,
        shots: int = 256,
        max_resample: int = 5,
    ) -> None:
        self.n_layers_atom = n_layers_atom
        self.n_layers_bond = n_layers_bond
        self.rng = np.random.default_rng(seed)
        self.shots = shots
        self.max_resample = max_resample
        self.env = FiveAtomMolEnv()

        qc, atom_params, bond_params = build_sqmg_circuit(n_layers_atom, n_layers_bond)
        self.base_circuit = qc
        self.atom_params = atom_params
        self.bond_params = bond_params
        self.sampler = sampler or Sampler()
        self._compiled = transpile(self.base_circuit, optimization_level=1)

        self.atom_weights = self.rng.normal(0.0, 0.2, size=len(atom_params))
        self.bond_weights = self.rng.normal(0.0, 0.2, size=len(bond_params))

    @property
    def num_weights(self) -> int:
        return len(self.atom_weights) + len(self.bond_weights)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.atom_weights, self.bond_weights])

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape[0] == self.num_weights
        self.atom_weights = np.array(new_w[: len(self.atom_weights)], copy=True)
        self.bond_weights = np.array(new_w[len(self.atom_weights) :], copy=True)

    def _bind(self) -> dict:
        bind = {}
        for i, w in enumerate(self.atom_weights):
            bind[self.atom_params[i]] = w
        for i, w in enumerate(self.bond_weights):
            bind[self.bond_params[i]] = w
        return bind

    def _bitstring_to_actions(self, bitstring: str) -> Tuple[List[int], List[int]]:
        bits = list(reversed(bitstring))
        atoms: List[int] = []
        bonds: List[int] = []

        for i in range(N_ATOMS):
            base = i * ATOM_Q
            val = (int(bits[base]) | (int(bits[base + 1]) << 1) | (int(bits[base + 2]) << 2))
            atoms.append(val)

        bond_offset = N_ATOMS * ATOM_Q
        for j in range(N_ATOMS - 1):
            base = bond_offset + j * BOND_Q
            val = (int(bits[base]) | (int(bits[base + 1]) << 1))
            bonds.append(val)
        return atoms, bonds

    def _decode_atoms(self, atom_states: List[int], resample_fn) -> List[int]:
        decoded: List[int] = []
        for st in atom_states:
            if st in ATOM_STATE_MAP:
                decoded.append(ATOM_VOCAB.index(ATOM_STATE_MAP[st]))
            else:
                decoded.append(-1)
        if all(x >= 0 for x in decoded):
            return decoded
        for _ in range(self.max_resample):
            atom_states, _ = resample_fn()
            decoded = []
            bad = False
            for st in atom_states:
                if st in ATOM_STATE_MAP:
                    decoded.append(ATOM_VOCAB.index(ATOM_STATE_MAP[st]))
                else:
                    bad = True
                    break
            if not bad:
                return decoded
        return [st % len(ATOM_VOCAB) for st in atom_states]

    def _decode_bonds(self, bond_states: List[int]) -> List[int]:
        return [int(b % len(BOND_VOCAB)) for b in bond_states]

    def _sample_once(self) -> Tuple[List[int], List[int]]:
        bind = self._bind()
        bound = self._compiled.assign_parameters(bind)
        result = self.sampler.run(bound, shots=self.shots).result()
        quasi = result.quasi_dists[0]
        best = max(quasi.items(), key=lambda kv: kv[1])[0]
        if isinstance(best, int):
            bitstring = format(best, f"0{self.base_circuit.num_qubits}b")
        else:
            bitstring = best
        return self._bitstring_to_actions(bitstring)

    def sample_actions(self, batch_size: int = 1) -> SampledBatch:
        atoms_batch = np.zeros((batch_size, 5), dtype=int)
        bonds_batch = np.zeros((batch_size, 4), dtype=int)
        smiles: List[Optional[str]] = []
        valids: List[bool] = []
        uniques: List[bool] = []

        for i in range(batch_size):
            def resample():
                return self._sample_once()

            atom_states, bond_states = self._sample_once()
            atom_ids = self._decode_atoms(atom_states, resample)
            bond_ids = self._decode_bonds(bond_states)

            atoms_batch[i] = atom_ids
            bonds_batch[i] = bond_ids
            before_seen = len(self.env.seen_smiles)
            smi, v = self.env.build_smiles_from_actions(atom_ids, bond_ids)
            after_seen = len(self.env.seen_smiles)
            smiles.append(smi)
            valids.append(v)
            uniques.append(bool(v and after_seen > before_seen))

        return SampledBatch(
            atoms=atoms_batch, bonds=bonds_batch, smiles=smiles, valids=valids, uniques=uniques
        )


__all__ = ["SQMGQiskitGenerator"]
