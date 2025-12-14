from __future__ import annotations

from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2


def build_sqmg_circuit(
    n_layers_atom: int = 2, n_layers_bond: int = 2
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    SQMG-style circuit with 3 qubits per atom (no reuse) and 2 qubits per bond (shared params reused).
    Total qubits = 15 (atoms) + 8 (bonds) = 23; all measured.
    """
    n_qubits = N_ATOMS * ATOM_Q + (N_ATOMS - 1) * BOND_Q
    qc = QuantumCircuit(n_qubits, n_qubits)

    atom_params = ParameterVector("atom", N_ATOMS * n_layers_atom * ATOM_Q * 2)
    bond_params = ParameterVector("bond", n_layers_bond * BOND_Q * 2)

    def atom_qubits(i_atom: int):
        base = i_atom * ATOM_Q
        return [base + j for j in range(ATOM_Q)]

    def bond_qubits(i_bond: int):
        base = N_ATOMS * ATOM_Q + i_bond * BOND_Q
        return [base + j for j in range(BOND_Q)]

    a_idx = 0
    for atom_idx in range(N_ATOMS):
        qs = atom_qubits(atom_idx)
        for _ in range(n_layers_atom):
            for q in qs:
                qc.ry(atom_params[a_idx], q)
                qc.rz(atom_params[a_idx + 1], q)
                a_idx += 2
            qc.cx(qs[0], qs[1])
            qc.cx(qs[1], qs[2])

    for bond_idx in range(N_ATOMS - 1):
        qs = bond_qubits(bond_idx)
        b_idx = 0
        for _ in range(n_layers_bond):
            for q in qs:
                qc.ry(bond_params[b_idx], q)
                qc.rz(bond_params[b_idx + 1], q)
                b_idx += 2
            qc.cx(qs[0], qs[1])

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, atom_params, bond_params


__all__ = ["build_sqmg_circuit", "N_ATOMS", "ATOM_Q", "BOND_Q"]
