from __future__ import annotations

from typing import List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2


def build_sqmg_chain_circuit(
    n_atoms: int = 5,
    atom_layers: int = 2,
    bond_layers: int = 1,
    seed: int | None = None,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build PDF-style SQMG/QCNC circuit (static atoms + dynamic bonds).

    Design (N=5):
    - Qubits: 3N + 2 = 17 (15 atom qubits + 2 bond qubits)
    - Classical bits: 3N + 2(N-1) = 23 (15 atom bits + 8 bond bits)

    Atom decode (3-bit):
      000 -> NONE
      001 -> C
      010 -> O
      011 -> N
      100..111 -> NONE

    Bond decode (2-bit):
      00 -> NONE
      01 -> SINGLE
      10 -> DOUBLE
      11 -> TRIPLE

    Bond module is applied only when both endpoint atom codes != 000 (non-NONE).
    Bond qubits are reset between bonds.
    """
    assert n_atoms == N_ATOMS, "This project is fixed to N=5"
    n_bonds = n_atoms - 1

    # 5 atom quantum registers (3 qubits each)
    q_atoms = [QuantumRegister(ATOM_Q, f"qa{i}") for i in range(n_atoms)]
    # 1 bond quantum register (2 qubits), reused
    q_bond = QuantumRegister(BOND_Q, "qb")

    # 5 atom classical registers (3 bits each)
    c_atoms = [ClassicalRegister(ATOM_Q, f"ca{i}") for i in range(n_atoms)]
    # 4 bond classical registers (2 bits each)
    c_bonds = [ClassicalRegister(BOND_Q, f"cb{i}") for i in range(n_bonds)]

    qc = QuantumCircuit(*q_atoms, q_bond, *c_atoms, *c_bonds)

    # parameters: per-atom (no reuse) + shared bond params (reuse)
    atom_params: List[Parameter] = []
    for i in range(n_atoms):
        for layer in range(atom_layers):
            for q in range(ATOM_Q):
                atom_params.append(Parameter(f"a_{i}_{layer}_{q}_ry"))
                atom_params.append(Parameter(f"a_{i}_{layer}_{q}_rz"))

    bond_params: List[Parameter] = []
    for layer in range(bond_layers):
        for q in range(BOND_Q):
            bond_params.append(Parameter(f"b_{layer}_{q}_ry"))
            bond_params.append(Parameter(f"b_{layer}_{q}_rz"))

    params = atom_params + bond_params

    # Atom blocks
    p_idx = 0
    for i in range(n_atoms):
        qs = q_atoms[i]
        for _ in range(atom_layers):
            for q in range(ATOM_Q):
                qc.ry(atom_params[p_idx], qs[q])
                qc.rz(atom_params[p_idx + 1], qs[q])
                p_idx += 2
            qc.cx(qs[0], qs[1])
            qc.cx(qs[1], qs[2])
        qc.measure(qs, c_atoms[i])

    # Bond blocks (dynamic + reuse)
    for b in range(n_bonds):
        qc.reset(q_bond)

        with qc.if_test((c_atoms[b], 0)) as else_left:
            pass
        with else_left:
            with qc.if_test((c_atoms[b + 1], 0)) as else_right:
                pass
            with else_right:
                # apply shared bond ansatz
                bp_idx = 0
                for _ in range(bond_layers):
                    for q in range(BOND_Q):
                        qc.ry(bond_params[bp_idx], q_bond[q])
                        qc.rz(bond_params[bp_idx + 1], q_bond[q])
                        bp_idx += 2
                    qc.cx(q_bond[0], q_bond[1])

        qc.measure(q_bond, c_bonds[b])
        qc.reset(q_bond)

    return qc, params


__all__ = ["build_sqmg_chain_circuit", "N_ATOMS", "ATOM_Q", "BOND_Q"]
