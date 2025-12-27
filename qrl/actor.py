"""Quantum actor PQC for A2C.

Constructor defaults: n_qubits=8 (recommended 6-8) and n_layers=2 (recommended 2-3).
"""

from __future__ import annotations

from typing import List, Tuple

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

import importlib


def _import_cudaq():  # pragma: no cover - optional dependency
    try:
        return importlib.import_module("cudaq")
    except ImportError:
        return None


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


def _z_expectations_from_counts(
    counts: dict[str, int], n_qubits: int, shots: int
) -> np.ndarray:
    total = max(shots, 1)
    z = np.zeros(n_qubits, dtype=float)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(n_qubits):
            bit = bits[-1 - q]
            z[q] += (1.0 if bit == "0" else -1.0) * count
    return z / total


def _softplus(x: float) -> float:
    if x >= 0:
        return float(x + np.log1p(np.exp(-x)))
    return float(np.log1p(np.exp(x)))


def gaussian_logprob(action: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    action = np.asarray(action, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if action.shape != mu.shape:
        raise ValueError("action and mu must have the same shape")
    sigma = float(max(sigma, 1e-6))
    diff = (action - mu) / sigma
    return float(-0.5 * np.sum(diff**2 + np.log(2.0 * np.pi * sigma**2)))


def gaussian_entropy(action_dim: int, sigma: float) -> float:
    sigma = float(max(sigma, 1e-6))
    return float(action_dim * (0.5 * np.log(2.0 * np.pi * np.e * sigma**2)))


class QiskitQuantumActor:
    """Quantum actor that outputs (mu, sigma) for a diagonal Gaussian policy."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        action_dim: int = 16,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        seed: int | None = None,
    ) -> None:
        if not _HAS_QISKIT:
            raise RuntimeError("qiskit is not installed; install requirements-cpu.txt")
        self.state_dim = int(state_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.action_dim = int(action_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
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
        self.proj_sigma = self.rng.normal(0.0, 1.0, size=(self.n_qubits,))
        self.proj_sigma /= np.sqrt(self.n_qubits)

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        angles = _state_to_angles(state, self.n_qubits)
        bind = {self.input_params[i]: float(angles[i]) for i in range(self.n_qubits)}
        for i, w in enumerate(self.weights):
            bind[self.weight_params[i]] = float(w)
        bound = self.base_circuit.assign_parameters(bind, inplace=False)
        sv = Statevector.from_instruction(bound)
        z = _z_expectations_from_statevector(sv, self.n_qubits)
        mu = np.tanh(self.proj @ z)
        log_sigma = float(self.proj_sigma @ z)
        sigma = _softplus(log_sigma)
        sigma = float(np.clip(sigma, self.sigma_min, self.sigma_max))
        return mu, sigma


class CudaQQuantumActor:
    """CUDA-Q actor that outputs (mu, sigma) for a diagonal Gaussian policy."""

    def __init__(
        self,
        state_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        action_dim: int = 16,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
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
        self.action_dim = int(action_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.shots = int(shots)
        self.rng = np.random.default_rng(seed)

        self.weights = self.rng.normal(0.0, 0.2, size=self.n_layers * self.n_qubits * 2)
        self.proj = self.rng.normal(0.0, 1.0, size=(self.action_dim, self.n_qubits))
        self.proj /= np.sqrt(self.n_qubits)
        self.proj_sigma = self.rng.normal(0.0, 1.0, size=(self.n_qubits,))
        self.proj_sigma /= np.sqrt(self.n_qubits)
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
            self.kernel.mz(q[qb], f"q{qb:02d}")

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
            counts = self._get_register_counts(result, f"q{qb:02d}")
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
            z[qb] = (zeros - ones) / total
        return z

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        angles = _state_to_angles(state, self.n_qubits)
        result = self._cudaq.sample(
            self.kernel, angles.tolist(), self.weights.tolist(), shots_count=self.shots
        )
        z = self._z_expectations_from_registers(result)
        mu = np.tanh(self.proj @ z)
        log_sigma = float(self.proj_sigma @ z)
        sigma = _softplus(log_sigma)
        sigma = float(np.clip(sigma, self.sigma_min, self.sigma_max))
        return mu, sigma


__all__ = [
    "QiskitQuantumActor",
    "CudaQQuantumActor",
    "gaussian_logprob",
    "gaussian_entropy",
]
