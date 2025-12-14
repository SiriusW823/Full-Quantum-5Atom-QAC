import numpy as np
from typing import Dict, Any
from rdkit import Chem
import gymnasium as gym
from gymnasium import spaces
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# 5-bit scaffold map (toy library)
BITSTRING_TO_SMILES: Dict[str, str] = {
    "00000": "CCCCC",
    "00001": "C1CCCC1",
    "00010": "C=CC=C",
    "00011": "C#CC#C",
    "00100": "NCCCN",
    "00101": "OCCCO",
    "00110": "CCNCC",
    "00111": "CCOCC",
    "01000": "CCNCN",
    "01001": "CCOCO",
    "01010": "CNCNC",
    "01011": "COCOC",
    "01100": "NCOCN",
    "01101": "NCONC",
    "01110": "OCOCO",
    "01111": "OCNCO",
    "10000": "C1CCN1",
    "10001": "C1CCO1",
    "10010": "N1CCN1",
    "10011": "O1CCO1",
    "10100": "C1NCO1",
    "10101": "C1OCO1",
    "10110": "N1COC1",
    "10111": "O1CNC1",
    "11000": "CCCNC",
    "11001": "CCCOC",
    "11010": "CCNOC",
    "11011": "CCONC",
    "11100": "NCCOC",
    "11101": "OCCNC",
    "11110": "NCCCN",
    "11111": "OCCCO",
}

STRUCTURE_NAMES = {0: "Linear", 1: "Full", 2: "Ring"}
NUM_STRUCTURES = 3
MAX_HEAVY = 5


class MoleculeGenEnv(gym.Env):
    """
    One-step quantum generator:
    - Action: Box(11,) -> 10 rotation params + 1 structure selector (continuous -> discrete 0..2)
    - Observation: dummy zero vector
    - Reward: 10 * (valid * unique) based on bitstring->SMILES mapping
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(11,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.sim = AerSimulator()
        self.seen: set[str] = set()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        params = np.asarray(action, dtype=np.float32).flatten()
        if params.shape[0] < 11:
            raise ValueError("Action must have length 11 (10 angles + 1 structure value).")
        rot_params = np.clip(params[:10], -np.pi, np.pi)
        structure_val = params[10]
        structure_idx = int(np.round(structure_val)) % NUM_STRUCTURES

        bitstring = self._run_qiskit_circuit(rot_params, structure_idx)
        smiles = BITSTRING_TO_SMILES.get(bitstring, None)

        valid = 0.0
        unique = 0.0
        if smiles and self._is_valid_smiles(smiles):
            valid = 1.0
            if smiles not in self.seen:
                unique = 1.0
                self.seen.add(smiles)

        # Reward shaping: small bonus for validity, big bonus for uniqueness, small penalty for invalid
        BASE_VALIDITY_REWARD = 0.01
        if valid >= 1.0:
            reward = BASE_VALIDITY_REWARD
            if unique >= 1.0:
                reward += 10.0 * unique
        else:
            reward = -0.1
        info = {"bitstring": bitstring, "smiles": smiles, "valid": valid, "unique": unique, "structure": STRUCTURE_NAMES[structure_idx]}
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), True, False, info

    def _run_qiskit_circuit(self, params: np.ndarray, structure_idx: int) -> str:
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.ry(float(params[i % 5]), i)
            qc.rz(float(params[5 + (i % 5)]), i)

        if structure_idx == 1:  # Full
            for i in range(5):
                for j in range(i + 1, 5):
                    qc.cx(i, j)
        elif structure_idx == 2:  # Ring
            for i in range(5):
                qc.cx(i, (i + 1) % 5)
        else:  # Linear
            for i in range(4):
                qc.cx(i, i + 1)

        qc.measure_all()
        compiled = transpile(qc, self.sim)
        # Increase shots and use majority vote to reduce randomness
        result = self.sim.run(compiled, shots=100).result()
        counts = result.get_counts()
        bitstring = max(counts, key=counts.get)
        bitstring = bitstring.zfill(5)[-5:]
        return bitstring[::-1]  # reverse bit order for map

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return False
            return mol.GetNumHeavyAtoms() <= MAX_HEAVY
        except Exception:
            return False
