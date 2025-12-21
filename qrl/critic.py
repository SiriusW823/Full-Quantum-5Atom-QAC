"""Quantum critic PQC for A2C (scalar value in [0, 1]).

Constructor defaults: n_qubits=8 and n_layers=2, both tunable.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from qrl.actor import _state_to_angles, _z_expectations_from_statevector


class QiskitQuantumCritic:
    """Quantum critic that outputs a scalar value in [0, 1]."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seed: int | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
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
        self.proj = self.rng.normal(0.0, 1.0, size=(self.n_qubits,))
        self.proj /= np.sqrt(self.n_qubits)

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def forward(self, state: np.ndarray) -> float:
        angles = _state_to_angles(state, self.n_qubits)
        bind = {self.input_params[i]: float(angles[i]) for i in range(self.n_qubits)}
        for i, w in enumerate(self.weights):
            bind[self.weight_params[i]] = float(w)
        bound = self.base_circuit.assign_parameters(bind, inplace=False)
        sv = Statevector.from_instruction(bound)
        z = _z_expectations_from_statevector(sv, self.n_qubits)
        raw = float(self.proj @ z)
        val = (np.tanh(raw) + 1.0) * 0.5
        return float(np.clip(val, 0.0, 1.0))


__all__ = ["QiskitQuantumCritic"]
