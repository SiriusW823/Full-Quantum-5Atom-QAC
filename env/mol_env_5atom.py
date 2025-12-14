from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")

# Vocabularies (exported)
ATOM_VOCAB: List[str] = ["C", "N", "O", "S", "F"]
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
    RDKit environment for an exact 5-atom chain (5 heavy atoms, 4 bonds).

    Atoms: indices into ATOM_VOCAB
    Bonds: indices into BOND_VOCAB
    Topology: atom0-atom1-atom2-atom3-atom4
    """

    def __init__(self, enforce_single_fragment: bool = True) -> None:
        self.metrics = Metrics()
        self.enforce_single_fragment = enforce_single_fragment
        self.seen_smiles: Set[str] = set()

    def reset(self) -> None:
        self.metrics = Metrics()
        self.seen_smiles.clear()

    def is_unique(self, smiles: str) -> bool:
        if smiles in self.seen_smiles:
            return False
        self.seen_smiles.add(smiles)
        return True

    def build_smiles_from_actions(
        self, atoms: Sequence[int], bonds: Sequence[int]
    ) -> Tuple[Optional[str], bool]:
        """
        Build SMILES from atom/bond indices. Updates counters on valid molecules.
        Returns (smiles|None, valid_flag).
        """
        self.metrics.samples += 1

        if len(atoms) != 5 or len(bonds) != 4:
            return None, False

        # Map vocab
        try:
            atom_syms = [ATOM_VOCAB[a] for a in atoms]
            bond_types = [_BOND_TYPE_MAP[BOND_VOCAB[b]] for b in bonds]
        except (IndexError, KeyError):
            return None, False

        mol = Chem.RWMol()
        atom_indices: List[int] = []

        try:
            for sym in atom_syms:
                atom_indices.append(mol.AddAtom(Chem.Atom(sym)))

            for i, bt in enumerate(bond_types):
                if bt is None:
                    continue  # "NONE" => no bond
                mol.AddBond(atom_indices[i], atom_indices[i + 1], bt)

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            if self.enforce_single_fragment and "." in smiles:
                return None, False

            self.metrics.valid_count += 1
            if self.is_unique(smiles):
                self.metrics.unique_valid_count += 1

            return smiles, True
        except Exception:
            return None, False


__all__ = ["ATOM_VOCAB", "BOND_VOCAB", "FiveAtomMolEnv", "Metrics"]
