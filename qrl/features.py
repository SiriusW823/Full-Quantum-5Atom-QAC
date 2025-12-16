from __future__ import annotations

from typing import Dict, List

import numpy as np
from rdkit import Chem


ATOM_TOKENS = ["C", "N", "O", "S", "F"]
BOND_TOKENS = ["SINGLE", "DOUBLE", "TRIPLE"]


def _safe_mol(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    return mol


def smiles_to_features(smiles: str, dim: int = 12) -> np.ndarray:
    """
    Map SMILES to a small fixed-length feature vector in [0, 1].
    Features (<=12 dims):
      0-4: atom counts for C,N,O,S,F
      5-7: bond counts for SINGLE, DOUBLE, TRIPLE
      8: ring count
      9: aromatic atom count
      remaining dims (if any): zeros
    """
    feats: List[float] = [0.0] * dim
    mol = _safe_mol(smiles)
    if mol is None:
        return np.zeros(dim, dtype=float)

    # Atom counts
    atom_counts: Dict[str, int] = {k: 0 for k in ATOM_TOKENS}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in atom_counts:
            atom_counts[sym] += 1
    for i, sym in enumerate(ATOM_TOKENS):
        feats[i] = atom_counts[sym]

    # Bond counts
    bond_counts: Dict[str, int] = {k: 0 for k in BOND_TOKENS}
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            bond_counts["SINGLE"] += 1
        elif bt == Chem.BondType.DOUBLE:
            bond_counts["DOUBLE"] += 1
        elif bt == Chem.BondType.TRIPLE:
            bond_counts["TRIPLE"] += 1
    feats[5] = bond_counts["SINGLE"]
    feats[6] = bond_counts["DOUBLE"]
    feats[7] = bond_counts["TRIPLE"]

    # Ring / aromatic
    # NOTE: In RDKit 2022.09.x, Chem.GetSSSR/GetSymmSSSR return ring atom index vectors,
    # not a numeric count. Use RingInfo.NumRings() for a stable scalar feature.
    feats[8] = float(mol.GetRingInfo().NumRings())
    feats[9] = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))

    # Normalize to [0,1] with clipping; rough scale by max 10
    arr = np.array(feats, dtype=float)
    arr = np.clip(arr / 10.0, 0.0, 1.0)
    return arr


__all__ = ["smiles_to_features", "ATOM_TOKENS", "BOND_TOKENS"]
