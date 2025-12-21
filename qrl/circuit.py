from __future__ import annotations

from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_qrl_pqc(
    n_qubits: int = 4, n_layers: int = 2
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Small PQC for QRL helper.
    Returns:
        circuit with measurements
        data_params: ParameterVector for encoded features (len = n_qubits)
        weight_params: trainable weights (len = n_layers * n_qubits * 2)
    """
    data_params = ParameterVector("f", n_qubits)
    weight_params = ParameterVector("theta", n_layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits, n_qubits)

    for i in range(n_qubits):
        qc.ry(data_params[i], i)
        qc.rz(data_params[i], i)

    w_idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(weight_params[w_idx], q)
            qc.rz(weight_params[w_idx + 1], q)
            w_idx += 2
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, data_params, weight_params


__all__ = ["build_qrl_pqc"]
