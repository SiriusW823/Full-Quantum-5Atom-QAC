"""Quantum critic PQC for A2C (scalar value in [0, 1]).

Constructor defaults: n_qubits=8 and n_layers=2, both tunable.
"""

from __future__ import annotations

import numpy as np
try:  # optional dependency
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    _HAS_QISKIT = True
except ImportError:  # pragma: no cover - optional dependency
    QuantumCircuit = None
    ParameterVector = None
    Statevector = None
    _HAS_QISKIT = False

from qrl.actor import _state_to_angles, _z_expectations_from_statevector

import importlib


def _import_cudaq():  # pragma: no cover - optional dependency
    try:
        return importlib.import_module("cudaq")
    except ImportError:
        return None


class QiskitQuantumCritic:
    """Quantum critic that outputs a scalar value in [0, 1]."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seed: int | None = None,
    ) -> None:
        if not _HAS_QISKIT:
            raise RuntimeError("qiskit is not installed; install requirements-cpu.txt")
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


class CudaQQuantumCritic:
    """CUDA-Q critic that outputs a scalar value in [0, 1]."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        shots: int = 256,
        seed: int | None = None,
    ) -> None:
        cudaq_mod = _import_cudaq()
        if cudaq_mod is None:
            raise RuntimeError("cudaq is not available")
        self._cudaq = cudaq_mod
        self.state_dim = int(state_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.shots = int(shots)
        self.rng = np.random.default_rng(seed)

        self.weights = self.rng.normal(0.0, 0.2, size=self.n_layers * self.n_qubits * 2)
        self.proj = self.rng.normal(0.0, 1.0, size=(self.n_qubits,))
        self.proj /= np.sqrt(self.n_qubits)
        self.kernel, self.input_params, self.weight_params = self._cudaq.make_kernel(
            list, list
        )
        q = self.kernel.qalloc(self.n_qubits)
        for i in range(self.n_qubits):
            self.kernel.ry(self.input_params[i], q[i])
            self.kernel.rz(self.input_params[i], q[i])
        w_idx = 0
        for _ in range(self.n_layers):
            for qb in range(self.n_qubits):
                self.kernel.ry(self.weight_params[w_idx], q[qb])
                self.kernel.rz(self.weight_params[w_idx + 1], q[qb])
                w_idx += 2
            for qb in range(self.n_qubits - 1):
                self.kernel.cx(q[qb], q[qb + 1])
            if self.n_qubits > 1:
                self.kernel.cx(q[self.n_qubits - 1], q[0])
        for qb in range(self.n_qubits):
            self.kernel.mz(q[qb], f"c{qb:02d}")

    def _get_register_counts(self, result, name: str) -> dict[str, int]:
        try:
            sub = result.get_register_counts(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read register counts: name={name}, result_type={type(result)}"
            ) from exc

        try:
            return {k: int(v) for k, v in sub.items()}
        except Exception:
            pass
        try:
            dct = dict(sub)
            return {k: int(v) for k, v in dct.items()}
        except Exception:
            pass
        try:
            keys = list(sub)
            out: dict[str, int] = {}
            for k in keys:
                out[k] = int(sub[k])
            return out
        except Exception as exc:
            raise RuntimeError(
                "Failed to read register counts: "
                f"name={name}, result_type={type(result)}, sub_type={type(sub)}"
            ) from exc

    def _z_expectations_from_registers(self, result) -> np.ndarray:
        z = np.zeros(self.n_qubits, dtype=float)
        total = max(self.shots, 1)
        for qb in range(self.n_qubits):
            counts = self._get_register_counts(result, f"c{qb:02d}")
            zeros = 0
            ones = 0
            for bitstring, count in counts.items():
                bits = str(bitstring).strip().replace(" ", "")
                if bits.startswith("0b"):
                    bits = bits[2:]
                bit = bits[-1] if bits else "0"
                if bit == "0":
                    zeros += count
                else:
                    ones += count
            z[qb] = (zeros - ones) / float(total)
        return z

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
        result = self._cudaq.sample(
            self.kernel, angles.tolist(), self.weights.tolist(), shots_count=self.shots
        )
        z = self._z_expectations_from_registers(result)
        raw = float(self.proj @ z)
        val = (np.tanh(raw) + 1.0) * 0.5
        return float(np.clip(val, 0.0, 1.0))


__all__ = ["QiskitQuantumCritic", "CudaQQuantumCritic"]
