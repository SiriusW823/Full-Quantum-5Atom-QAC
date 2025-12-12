import numpy as np
from typing import Dict, Any, List, Tuple
from rdkit import Chem
import gymnasium as gym
from gymnasium import spaces

# Token mapping for sequential SMILES generation
TOKENS = ["C", "N", "O", "=", "#", "(", ")", "1", "2", "STOP"]
TOKEN_STOP_IDX = 9
MAX_LEN = 50
MAX_HEAVY = 5


class MoleculeGenEnv(gym.Env):
    """
    Sequential SMILES generation environment.
    - Action: Discrete(10) tokens to append; 9 = STOP.
    - Observation: length-50 int array of token indices, padded with -1.
    - Reward: 0 for intermediate steps; terminal reward = 1.0 if SMILES is valid
      and heavy atoms <= 5, else 0.0.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(TOKENS))
        self.observation_space = spaces.Box(low=-1, high=len(TOKENS) - 1, shape=(MAX_LEN,), dtype=np.int32)
        self.history: List[int] = []
        self.done = False

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.history = []
        self.done = False
        return self._get_obs(), {}

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        a = int(np.clip(action, 0, len(TOKENS) - 1))
        self.history.append(a)

        terminated = (a == TOKEN_STOP_IDX) or (len(self.history) >= MAX_LEN)
        reward = 0.0
        info: Dict[str, Any] = {}

        if terminated:
            smiles, valid = self.finalize()
            reward = 1.0 if valid else 0.0
            info = {
                "smiles": smiles,
                "valid": float(valid),
            }
            self.done = True

        return self._get_obs(), float(reward), self.done, False, info

    def _get_obs(self) -> np.ndarray:
        obs = self.history + [-1] * (MAX_LEN - len(self.history))
        return np.array(obs, dtype=np.int32)

    def finalize(self) -> Tuple[str | None, bool]:
        # Remove trailing STOP
        tokens = [TOKENS[i] for i in self.history if i != TOKEN_STOP_IDX]
        if not tokens:
            return None, False
        smiles = "".join(tokens)
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return smiles, False
            if mol.GetNumHeavyAtoms() > MAX_HEAVY:
                return smiles, False
            return smiles, True
        except Exception:
            return smiles, False
