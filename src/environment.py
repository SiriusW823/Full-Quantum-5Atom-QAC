import numpy as np
from typing import Dict, Any, Tuple
from rdkit import Chem
import gymnasium as gym
from gymnasium import spaces

# One-step bandit-like MDP:
# - Action: choose one of 4 predefined SMILES templates
# - Observation: dummy zero vector (shape=(1,))
# - Reward: 10 * (validity * uniqueness), uniqueness tracked across episodes

ACTION_SMILES = ["C", "N", "O", "CO"]
MAX_HEAVY = 5


class MoleculeGenEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTION_SMILES))
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.done = False
        self.seen: set[str] = set()
        self.last_smiles: str | None = None

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.done = False
        self.last_smiles = None
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        if self.done:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        a = int(np.clip(action, 0, len(ACTION_SMILES) - 1))
        smiles = ACTION_SMILES[a]
        valid = 1.0 if self._is_valid(smiles) else 0.0
        unique = 0.0
        if valid and smiles not in self.seen:
            unique = 1.0
            self.seen.add(smiles)
        reward = 10.0 * (valid * unique)

        self.done = True
        self.last_smiles = smiles
        info = {"smiles": smiles, "valid": valid, "unique": unique}
        return np.zeros(self.observation_space.shape, dtype=np.float32), float(reward), True, False, info

    def _is_valid(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return False
            return mol.GetNumHeavyAtoms() <= MAX_HEAVY
        except Exception:
            return False
