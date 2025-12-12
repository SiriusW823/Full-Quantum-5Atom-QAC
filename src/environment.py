import numpy as np
from typing import List, Tuple, Dict, Any
from rdkit import Chem
import gymnasium as gym
from gymnasium import spaces

ATOM_TYPES = ["NONE", "C", "N", "O"]
BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
MAX_ATOMS = 5
SEQUENCE_LENGTH = 9  # Atom1 -> Bond1 -> Atom2 -> Bond2 -> Atom3 -> Bond3 -> Atom4 -> Bond4 -> Atom5
C_VALIDITY_BONUS = 0.1


class MoleculeGenEnv(gym.Env):
    """
    Gym-compatible environment for linear 5-atom molecule generation.
    Observation: length-9 vector of token ids (0-3), padded with 0.
    Action space: Discrete(4) over token ids for current position.
    Reward: at terminal step only, (valid * unique) + 0.1 * valid.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=(SEQUENCE_LENGTH,), dtype=np.int64)
        self.history: List[int] = []
        self.done = False
        self.seen_smiles: set[str] = set()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.history = []
        self.done = False
        return self._get_obs(), {}

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        safe_action = int(np.clip(action, 0, 3))
        # Prevent trivial empty molecule: first atom cannot be NONE
        if len(self.history) == 0 and safe_action == 0:
            safe_action = 1

        self.history.append(safe_action)
        self.done = self._should_stop(safe_action)

        reward = 0.0
        info: Dict[str, Any] = {}

        if self.done:
            smiles, valid, unique = self.finalize()
            reward = self.shaped_reward(valid, unique)
            info = {
                "smiles": smiles,
                "valid": valid,
                "unique": unique,
                "length": len(self.history),
            }

        return self._get_obs(), float(reward), self.done, False, info

    def _get_obs(self) -> np.ndarray:
        obs = self.history + [0] * (SEQUENCE_LENGTH - len(self.history))
        return np.array(obs, dtype=np.int64)

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
        unique = 0.0
        if valid and smiles:
            unique = 1.0 if smiles not in self.seen_smiles else 0.0
            if unique:
                self.seen_smiles.add(smiles)
        return smiles, float(valid), float(unique)

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
