from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")

# Vocabularies (exported)
# Atom decode (3-bit): 000->NONE, 001->C, 010->O, 011->N, 100..111->NONE
ATOM_VOCAB: List[str] = ["NONE", "C", "O", "N"]
# Bond decode (2-bit): 00->NONE, 01->SINGLE, 10->DOUBLE, 11->TRIPLE
BOND_VOCAB: List[str] = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]

# Full-graph edges (unordered pairs), deterministic order.
EDGE_LIST: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]

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
    """Classical RDKit environment for 5 sites with optional NONE atoms and full-graph bonds.

    API:
      build_smiles_from_actions(atoms, bonds) -> (smiles|None, valid)

    Inputs:
      - atoms: length 5, indices into ATOM_VOCAB
      - bonds: length 10, indices into BOND_VOCAB, aligned with EDGE_LIST

    Validity:
      - active_len = count(non-NONE atoms) must be >= 2
      - add only non-NONE atoms to RDKit
      - add only bonds where both endpoints exist and bond != NONE
      - sanitize, canonicalize
      - reject fragments if enforce_single_fragment=True (SMILES contains '.')

    Uniqueness:
      - counted only when valid and canonical SMILES is new
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

        if len(atoms) != 5 or len(bonds) != len(EDGE_LIST):
            return None, False

        try:
            atom_syms = [ATOM_VOCAB[a] for a in atoms]
            bond_types = [_BOND_TYPE_MAP[BOND_VOCAB[b]] for b in bonds]
        except (IndexError, KeyError):
            return None, False

        active_sites = [i for i, sym in enumerate(atom_syms) if sym != "NONE"]
        if len(active_sites) < 2:
            return None, False

        mol = Chem.RWMol()
        site_to_rd: List[Optional[int]] = [None] * 5

        try:
            for site_idx, sym in enumerate(atom_syms):
                if sym == "NONE":
                    continue
                site_to_rd[site_idx] = mol.AddAtom(Chem.Atom(sym))

            for (i, j), bt in zip(EDGE_LIST, bond_types, strict=True):
                if bt is None:
                    continue
                a = site_to_rd[i]
                b = site_to_rd[j]
                if a is None or b is None:
                    continue
                mol.AddBond(a, b, bt)

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


__all__ = ["ATOM_VOCAB", "BOND_VOCAB", "EDGE_LIST", "FiveAtomMolEnv", "Metrics"]
