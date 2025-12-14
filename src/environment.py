import numpy as np
from typing import Dict, Any
from rdkit import Chem
import gymnasium as gym
from gymnasium import spaces
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

MAX_HEAVY = 5
N_QUBITS = 5
LAYERS = 3  # each layer: per-qubit Ry/Rz plus ring entanglement
PARAMS_PER_QUBIT_PER_LAYER = 2
ACTION_DIM = N_QUBITS * PARAMS_PER_QUBIT_PER_LAYER * LAYERS  # 5*2*3 = 30


class MoleculeGenEnv(gym.Env):
    """
    One-step quantum generator with direct atom mapping:
    - Action: Box(ACTION_DIM,) rotation params.
    - Circuit: Ry/Rz per qubit across multiple layers, ring entanglement each layer.
    - Measurement: 100 shots, majority vote bitstring.
    - Mapping: bit 0 -> C, bit 1 -> N; build a 5-atom ring.
    - Reward:
        invalid -> -0.1
        valid & repeated -> -0.5
        valid & unique -> +10.0
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.sim = AerSimulator()
        self.seen: set[str] = set()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        params = np.asarray(action, dtype=np.float32).flatten()
        if params.shape[0] < ACTION_DIM:
            raise ValueError(f"Action must have length {ACTION_DIM} rotation parameters.")
        params = np.clip(params, -np.pi, np.pi)

        bitstring = self._run_qiskit_circuit(params)
        smiles = self._bitstring_to_smiles(bitstring)

        valid = 0.0
        unique = 0.0
        if smiles and self._is_valid_smiles(smiles):
            valid = 1.0
            if smiles not in self.seen:
                unique = 1.0
                self.seen.add(smiles)

        if valid < 1.0:
            reward = -0.1
        elif unique < 1.0:
            reward = -0.5
        else:
            reward = 10.0

        info = {"bitstring": bitstring, "smiles": smiles, "valid": valid, "unique": unique}
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), True, False, info

    def _run_qiskit_circuit(self, params: np.ndarray) -> str:
        qc = QuantumCircuit(N_QUBITS)
        idx = 0
        for _ in range(LAYERS):
            for q in range(N_QUBITS):
                qc.ry(float(params[idx]), q)
                qc.rz(float(params[idx + 1]), q)
                idx += 2
            # ring entanglement
            for q in range(N_QUBITS):
                qc.cx(q, (q + 1) % N_QUBITS)

        qc.measure_all()
        compiled = transpile(qc, self.sim)
        result = self.sim.run(compiled, shots=100).result()
        counts = result.get_counts()
        bitstring = max(counts, key=counts.get)
        bitstring = bitstring.zfill(N_QUBITS)[-N_QUBITS:]
        return bitstring[::-1]  # reverse bit order

    @staticmethod
    def _bitstring_to_smiles(bitstring: str) -> str | None:
        atoms = ["N" if b == "1" else "C" for b in bitstring]
        if len(atoms) != N_QUBITS:
            return None
        mol = Chem.RWMol()
        atom_idx = [mol.AddAtom(Chem.Atom(sym)) for sym in atoms]
        # ring
        for i in range(N_QUBITS):
            mol.AddBond(atom_idx[i], atom_idx[(i + 1) % N_QUBITS], Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return False
            return mol.GetNumHeavyAtoms() <= MAX_HEAVY
        except Exception:
            return False
