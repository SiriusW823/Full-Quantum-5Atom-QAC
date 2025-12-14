from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_pqc(n_qubits: int = 4, n_layers: int = 2) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Build a modest PQC with data re-uploading and ring entanglement.

    Returns:
        circuit: parameterized QuantumCircuit without measurements
        data_params: ParameterVector for classical inputs (len = n_qubits)
        weight_params: ParameterVector for trainable weights (len = n_layers * n_qubits * 2)
    """
    data_params = ParameterVector("x", n_qubits)
    weight_params = ParameterVector("w", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)

    # Initial data encoding
    for i in range(n_qubits):
        qc.ry(data_params[i], i)
        qc.rz(data_params[i], i)

    # Variational layers with ring entanglement
    w_idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(weight_params[w_idx], q)
            qc.rz(weight_params[w_idx + 1], q)
            w_idx += 2
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)

    return qc, data_params, weight_params


def encode_inputs(head_id: int, noise: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Map head id and noise vector into n_qubits rotation angles.
    head id is scaled into [-pi, pi]; noise is clipped to a reasonable range.
    """
    vals = np.zeros(n_qubits, dtype=float)
    vals[0] = (float(head_id) % 8) / 7.0 * np.pi  # scale head id
    if noise.size:
        clipped = np.clip(noise, -3.0, 3.0)
        for i in range(1, min(n_qubits, clipped.size + 1)):
            vals[i] = clipped[i - 1]
    return vals
