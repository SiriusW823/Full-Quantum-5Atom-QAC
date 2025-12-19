from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def _state_to_angles(state: np.ndarray, n_qubits: int) -> np.ndarray:
    state = np.asarray(state, dtype=float).flatten()
    if state.size == 0:
        return np.zeros(n_qubits, dtype=float)
    if state.size >= n_qubits:
        chunks = np.array_split(state, n_qubits)
        pooled = np.array([float(c.mean()) for c in chunks], dtype=float)
    else:
        pooled = np.pad(state, (0, n_qubits - state.size), mode="constant")
    pooled = np.clip(pooled, -1.0, 1.0)
    return pooled * np.pi


def _z_expectations_from_statevector(statevector: Statevector, n_qubits: int) -> np.ndarray:
    probs = np.abs(statevector.data) ** 2
    z = np.zeros(n_qubits, dtype=float)
    for idx, p in enumerate(probs):
        for q in range(n_qubits):
            if (idx >> q) & 1:
                z[q] -= p
            else:
                z[q] += p
    return z


class QiskitQuantumActor:
    """Quantum actor that outputs a mean action vector (mu) in [-1, 1]."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        action_dim: int = 16,
        seed: int | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        self.input_params = ParameterVector("x", self.n_qubits)
        self.weight_params = ParameterVector("w", self.n_layers * self.n_qubits * 2)

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(self.input_params[i], i)
            qc.rz(self.input_params[i], i)

        w_idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(self.weight_params[w_idx], q)
                qc.rz(self.weight_params[w_idx + 1], q)
                w_idx += 2
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)

        self.base_circuit = qc
        self.weights = self.rng.normal(0.0, 0.2, size=len(self.weight_params))
        self.proj = self.rng.normal(0.0, 1.0, size=(self.action_dim, self.n_qubits))
        self.proj /= np.sqrt(self.n_qubits)

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def forward(self, state: np.ndarray) -> np.ndarray:
        angles = _state_to_angles(state, self.n_qubits)
        bind = {self.input_params[i]: float(angles[i]) for i in range(self.n_qubits)}
        for i, w in enumerate(self.weights):
            bind[self.weight_params[i]] = float(w)
        bound = self.base_circuit.assign_parameters(bind, inplace=False)
        sv = Statevector.from_instruction(bound)
        z = _z_expectations_from_statevector(sv, self.n_qubits)
        mu = np.tanh(self.proj @ z)
        return mu


__all__ = ["QiskitQuantumActor"]

