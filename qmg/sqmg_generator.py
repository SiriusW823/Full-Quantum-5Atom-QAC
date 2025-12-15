from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from env import ATOM_VOCAB, BOND_VOCAB, FiveAtomMolEnv
from qmg.generator import SampledBatch
from qmg.sqmg_circuit import N_ATOMS, build_sqmg_chain_circuit


# Atom 3-bit code mapping (as requested):
# 000 -> NONE
# 001 -> C
# 010 -> O
# 011 -> N
# 100..111 -> NONE
_ATOM_CODE_TO_ID = {
    0: 0,  # NONE
    1: 1,  # C
    2: 2,  # O
    3: 3,  # N
}


class SQMGQiskitGenerator:
    """PDF-style SQMG generator using AerSimulator (dynamic bonds, bond reuse)."""

    def __init__(
        self,
        atom_layers: int = 2,
        bond_layers: int = 1,
        shots: int = 256,
        seed: int | None = None,
    ) -> None:
        self.atom_layers = atom_layers
        self.bond_layers = bond_layers
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.env = FiveAtomMolEnv()

        qc, params = build_sqmg_chain_circuit(
            n_atoms=N_ATOMS, atom_layers=atom_layers, bond_layers=bond_layers
        )
        self.base_circuit = qc
        self.params = params

        self.weights = self.rng.normal(0.0, 0.2, size=len(self.params))

        self.backend = AerSimulator(seed_simulator=seed)
        self._compiled = transpile(self.base_circuit, self.backend, optimization_level=1)

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def _bind(self):
        bind = {p: float(self.weights[i]) for i, p in enumerate(self.params)}
        return self._compiled.assign_parameters(bind)

    def _parse_memory(self, mem: str) -> Dict[str, str]:
        """Return mapping creg_name -> bitstring for that register."""
        parts = mem.split()
        cregs = list(self.base_circuit.cregs)
        if len(parts) == len(cregs):
            return {creg.name: part for creg, part in zip(reversed(cregs), parts)}

        # no spaces, fallback: slice by creg sizes (reverse order)
        flat = mem.replace(" ", "")
        out: Dict[str, str] = {}
        idx = 0
        for creg in reversed(cregs):
            n = creg.size
            out[creg.name] = flat[idx : idx + n]
            idx += n
        return out

    def _decode_shot(self, mem: str) -> Tuple[List[int], List[int]]:
        reg_bits = self._parse_memory(mem)

        atom_ids: List[int] = []
        for i in range(N_ATOMS):
            bits = reg_bits.get(f"ca{i}", "000")
            code = int(bits, 2)
            atom_ids.append(_ATOM_CODE_TO_ID.get(code, 0))

        bond_ids: List[int] = []
        for j in range(N_ATOMS - 1):
            bits = reg_bits.get(f"cb{j}", "00")
            code = int(bits, 2)
            bond_ids.append(int(code % len(BOND_VOCAB)))

        return atom_ids, bond_ids

    def sample_actions(self, batch_size: int = 1) -> SampledBatch:
        atoms_batch = np.zeros((batch_size, 5), dtype=int)
        bonds_batch = np.zeros((batch_size, 4), dtype=int)
        smiles: List[Optional[str]] = []
        valids: List[bool] = []
        uniques: List[bool] = []

        bound = self._bind()
        job = self.backend.run(bound, shots=batch_size, memory=True)
        result = job.result()
        memory = result.get_memory(0)

        for i, mem in enumerate(memory):
            atom_ids, bond_ids = self._decode_shot(mem)
            atoms_batch[i] = atom_ids
            bonds_batch[i] = bond_ids

            before = len(self.env.seen_smiles)
            smi, v = self.env.build_smiles_from_actions(atom_ids, bond_ids)
            after = len(self.env.seen_smiles)
            smiles.append(smi)
            valids.append(v)
            uniques.append(bool(v and after > before))

        return SampledBatch(
            atoms=atoms_batch, bonds=bonds_batch, smiles=smiles, valids=valids, uniques=uniques
        )


__all__ = ["SQMGQiskitGenerator"]
