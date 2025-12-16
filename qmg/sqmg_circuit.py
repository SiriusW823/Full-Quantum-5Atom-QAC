from __future__ import annotations

from typing import List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

# PDF-aligned constants for N=5
N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2

# Full-graph edges (unordered pairs), deterministic order.
EDGE_LIST: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]


def build_sqmg_hybrid_fullgraph_circuit(
    n_atoms: int = N_ATOMS,
    atom_layers: int = 2,
    bond_layers: int = 1,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build a PDF-like SQMG/QCNC hybrid circuit for N=5 with full-graph bonds.

    Key properties (Version 2):
    - Total qubits: 3N + 2 = 17 (15 atom qubits + 2 reused bond qubits)
    - No mid-circuit classical conditionals (no if_test/c_if). Conditional bond behavior is
      implemented in decoding: if either endpoint atom is NONE -> force bond to NONE.
    - Full-graph bonds: generate 10 bonds for all unordered pairs (i<j), aligned with EDGE_LIST.
    - Classical bits:
        atoms: 5 registers x 3 bits = 15 bits
        bonds: 10 registers x 2 bits = 20 bits
        total = 35 bits

    Atom decode mapping (3-bit code):
      000 -> NONE
      001 -> C
      010 -> O
      011 -> N
      100..111 -> NONE

    Bond decode mapping (2-bit code):
      00 -> NONE
      01 -> SINGLE
      10 -> DOUBLE
      11 -> TRIPLE
    """
    assert n_atoms == N_ATOMS, "This project is fixed to N=5 sites"

    # Atom quantum registers (3 qubits each, no reuse)
    q_atoms = [QuantumRegister(ATOM_Q, f"qa{i}") for i in range(n_atoms)]
    # Reused bond quantum register (2 qubits)
    q_bond = QuantumRegister(BOND_Q, "qb")

    # Atom classical registers (3 bits each)
    c_atoms = [ClassicalRegister(ATOM_Q, f"ca{i}") for i in range(n_atoms)]
    # Bond classical registers (2 bits each) for all edges
    c_bonds = [ClassicalRegister(BOND_Q, f"cb{k}") for k in range(len(EDGE_LIST))]

    qc = QuantumCircuit(*q_atoms, q_bond, *c_atoms, *c_bonds)

    # Trainable parameters: per-atom (no reuse) + shared bond params (reuse across all edges)
    atom_params: List[Parameter] = []
    for atom_idx in range(n_atoms):
        for layer in range(atom_layers):
            for q in range(ATOM_Q):
                atom_params.append(Parameter(f"a_{atom_idx}_{layer}_{q}_ry"))
                atom_params.append(Parameter(f"a_{atom_idx}_{layer}_{q}_rz"))

    bond_params: List[Parameter] = []
    for layer in range(bond_layers):
        for q in range(BOND_Q):
            bond_params.append(Parameter(f"b_{layer}_{q}_ry"))
            bond_params.append(Parameter(f"b_{layer}_{q}_rz"))

    params = atom_params + bond_params

    # Atom blocks (independent)
    p_idx = 0
    for atom_idx in range(n_atoms):
        qs = q_atoms[atom_idx]
        for _ in range(atom_layers):
            for q in range(ATOM_Q):
                qc.ry(atom_params[p_idx], qs[q])
                qc.rz(atom_params[p_idx + 1], qs[q])
                p_idx += 2
            qc.cx(qs[0], qs[1])
            qc.cx(qs[1], qs[2])
        qc.measure(qs, c_atoms[atom_idx])

    # Bond blocks (reuse qb across all full-graph edges)
    for edge_idx in range(len(EDGE_LIST)):
        qc.reset(q_bond)
        bp_idx = 0
        for _ in range(bond_layers):
            for q in range(BOND_Q):
                qc.ry(bond_params[bp_idx], q_bond[q])
                qc.rz(bond_params[bp_idx + 1], q_bond[q])
                bp_idx += 2
            qc.cx(q_bond[0], q_bond[1])
        qc.measure(q_bond, c_bonds[edge_idx])
        qc.reset(q_bond)

    return qc, params


__all__ = [
    "ATOM_Q",
    "BOND_Q",
    "EDGE_LIST",
    "N_ATOMS",
    "build_sqmg_hybrid_fullgraph_circuit",
]

