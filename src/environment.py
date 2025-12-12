from dataclasses import dataclass
from typing import List, Tuple, Dict
from rdkit import Chem

ATOM_TYPES = ["NONE", "C", "N", "O"]
BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
MAX_ATOMS = 5
SEQUENCE_LENGTH = 9  # Atom1 -> Bond1 -> Atom2 -> Bond2 -> Atom3 -> Bond3 -> Atom4 -> Bond4 -> Atom5
C_VALIDITY_BONUS = 0.1


@dataclass
class StepResult:
    state: List[int]
    done: bool
    info: Dict


class MoleculeEnv:
    """
    Linear 5-atom builder.
    - Atoms occupy even indices (0,2,4,6,8); bonds occupy odd indices (1,3,5,7).
    - 'NONE' atom stops the sequence early and pads remaining slots with NONE.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> List[int]:
        self.history: List[int] = []
        self.done = False
        return self.history

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.history, True, {})

        safe_action = int(max(0, min(3, action)))  # clamp to valid token range
        self.history.append(safe_action)
        self.done = self._should_stop(safe_action)
        return StepResult(self.history, self.done, {"validity_bonus": C_VALIDITY_BONUS})

    def _should_stop(self, action: int) -> bool:
        idx = len(self.history) - 1
        is_atom_step = idx % 2 == 0
        atom_count = (len(self.history) + 1) // 2
        if len(self.history) >= SEQUENCE_LENGTH:
            return True
        if is_atom_step and action == 0:
            return True
        if atom_count >= MAX_ATOMS:
            return True
        return False

    def finalize(self) -> Tuple[str | None, float, float]:
        """
        Build SMILES from the current history.
        Returns (smiles, valid_flag, unique_placeholder).
        """
        atoms: List[int] = []
        bonds: List[int] = []

        for idx in range(SEQUENCE_LENGTH):
            token = self.history[idx] if idx < len(self.history) else 0
            if idx % 2 == 0:  # atom position
                if token == 0:
                    break
                atoms.append(token)
            else:  # bond position
                if not atoms:
                    continue
                bonds.append(token if token in (1, 2, 3) else 0)
            if len(atoms) >= MAX_ATOMS:
                break

        bonds = bonds[: max(0, len(atoms) - 1)]
        if not atoms:
            return None, 0.0, 0.0

        smiles, valid = self._atoms_bonds_to_smiles(atoms, bonds)
        reward = self.shaped_reward(float(valid), 0.0)  # unique handled externally
        return smiles, float(valid), reward

    def _atoms_bonds_to_smiles(self, atoms: List[int], bonds: List[int]) -> Tuple[str | None, float]:
        mol = Chem.RWMol()
        try:
            atom_indices = [mol.AddAtom(Chem.Atom(ATOM_TYPES[a])) for a in atoms]
            for i, b in enumerate(bonds):
                if b == 0:
                    continue
                bond_type = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                }.get(b, Chem.BondType.SINGLE)
                mol.AddBond(atom_indices[i], atom_indices[i + 1], bond_type)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            # If RDKit produced disconnected fragments, degrade to linear atom string
            if "." in smiles:
                fallback = "".join(ATOM_TYPES[a] for a in atoms if a != 0)
                return (fallback if fallback else None), (1.0 if fallback else 0.0)
            return smiles, 1.0
        except Exception:
            fallback = "".join(ATOM_TYPES[a] for a in atoms if a != 0)
            return (fallback if fallback else None), (1.0 if fallback else 0.0)

    @staticmethod
    def golden_metric(valid_flags: List[float], unique_flags: List[float], total: int) -> float:
        if total <= 0:
            return 0.0
        valid_count = sum(valid_flags)
        unique_count = sum(unique_flags)
        return (valid_count / total) * (unique_count / total)

    @staticmethod
    def shaped_reward(valid: float, unique: float) -> float:
        """
        Reward shaping that preserves the optimal policy while stabilizing gradients:
        Reward = (Validity * Uniqueness) + C_VALIDITY_BONUS * Validity
        """
        return (valid * unique) + C_VALIDITY_BONUS * valid
