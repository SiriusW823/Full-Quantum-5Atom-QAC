from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")

# Vocabularies (exported)
# NOTE: Order is required by SQMG decoding/tests.
ATOM_VOCAB: List[str] = ["NONE", "C", "O", "N"]
BOND_VOCAB: List[str] = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]

_BOND_TYPE_MAP = {
    "NONE": None,
    "SINGLE": rdchem.BondType.SINGLE,
    "DOUBLE": rdchem.BondType.DOUBLE,
    "TRIPLE": rdchem.BondType.TRIPLE,
}


@dataclass
class Metrics:
    samples: int = 0
    valid_count: int = 0
    unique_valid_count: int = 0

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.samples if self.samples else 0.0

    @property
    def unique_ratio(self) -> float:
        return self.unique_valid_count / self.samples if self.samples else 0.0

    @property
    def target_metric(self) -> float:
        return self.valid_ratio * self.unique_ratio


class FiveAtomMolEnv:
    """
    RDKit environment for a 5-position chain with optional NONE atoms.

    - atoms: length 5, indices into ATOM_VOCAB (includes NONE)
    - bonds: length 4, indices into BOND_VOCAB

    Fragment-avoidance rule:
    - Only a contiguous prefix of non-NONE atoms is allowed.
      Once NONE appears, all later atoms must be NONE.
    - active atom count must be >= 2 (otherwise invalid)

    Molecule construction:
    - Build only the active prefix atoms.
    - Apply only the first (active_len-1) bond decisions.
    - RDKit sanitization is used; fragments ('.') are rejected.
    """

    def __init__(self, enforce_single_fragment: bool = True) -> None:
        self.metrics = Metrics()
        self.samples = 0
        self.valid_count = 0
        self.unique_valid_count = 0
        self.enforce_single_fragment = enforce_single_fragment
        self.seen_smiles: Set[str] = set()

    def reset(self) -> None:
        self.metrics = Metrics()
        self.samples = 0
        self.valid_count = 0
        self.unique_valid_count = 0
        self.seen_smiles.clear()

    def is_unique(self, smiles: str) -> bool:
        if smiles in self.seen_smiles:
            return False
        self.seen_smiles.add(smiles)
        return True

    def build_smiles_from_actions(
        self, atoms: Sequence[int], bonds: Sequence[int]
    ) -> Tuple[Optional[str], bool]:
        self.metrics.samples += 1
        self.samples = self.metrics.samples

        if len(atoms) != 5 or len(bonds) != 4:
            return None, False

        try:
            atom_syms_full = [ATOM_VOCAB[a] for a in atoms]
            bond_types = [_BOND_TYPE_MAP[BOND_VOCAB[b]] for b in bonds]
        except (IndexError, KeyError):
            return None, False

        active_len = 0
        for sym in atom_syms_full:
            if sym == "NONE":
                break
            active_len += 1

        if active_len < 2:
            return None, False

        if any(sym != "NONE" for sym in atom_syms_full[active_len:]):
            return None, False

        atom_syms = atom_syms_full[:active_len]

        mol = Chem.RWMol()
        atom_indices: List[int] = []

        try:
            for sym in atom_syms:
                atom_indices.append(mol.AddAtom(Chem.Atom(sym)))

            for i, bt in enumerate(bond_types[: active_len - 1]):
                if bt is None:
                    continue
                mol.AddBond(atom_indices[i], atom_indices[i + 1], bt)

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            if self.enforce_single_fragment and "." in smiles:
                return None, False

            self.metrics.valid_count += 1
            self.valid_count = self.metrics.valid_count
            if self.is_unique(smiles):
                self.metrics.unique_valid_count += 1
                self.unique_valid_count = self.metrics.unique_valid_count

            return smiles, True
        except Exception:
            return None, False

    @property
    def valid_ratio(self) -> float:
        s = int(getattr(self, "samples", 0) or 0)
        v = int(getattr(self, "valid_count", 0) or 0)
        return (v / s) if s > 0 else 0.0

    @property
    def unique_ratio(self) -> float:
        s = int(getattr(self, "samples", 0) or 0)
        u = int(getattr(self, "unique_valid_count", 0) or 0)
        return (u / s) if s > 0 else 0.0

    @property
    def target_metric(self) -> float:
        return self.valid_ratio * self.unique_ratio

    def stats(self) -> dict:
        return {
            "samples": int(getattr(self, "samples", 0) or 0),
            "valid_count": int(getattr(self, "valid_count", 0) or 0),
            "unique_valid_count": int(getattr(self, "unique_valid_count", 0) or 0),
            "valid_ratio": self.valid_ratio,
            "unique_ratio": self.unique_ratio,
            "target_metric": self.target_metric,
        }


__all__ = ["ATOM_VOCAB", "BOND_VOCAB", "FiveAtomMolEnv", "Metrics"]
