from __future__ import annotations

from typing import List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

# N is fixed to 5 for this project
N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2

# PDF chain bonds only (not full-graph)
EDGE_LIST: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4)]


def _compute_none_flag(qc: QuantumCircuit, atom_qubits, anc_qubit) -> None:
    """Compute anc_qubit = 1 iff atom code decodes to NONE (000).

    Decode mapping is fixed:
      000 -> NONE
      001 -> C
      010 -> O
      011 -> N
      100 -> S
      101 -> P
      110 -> F
      111 -> Cl

    Required reversible logic:
    - X on atom qubits (000 -> 111)
    - mcx with 3 controls -> ancilla
    - X back

    This routine is its own inverse, so calling it twice uncomputes.
    """
    for q in atom_qubits:
        qc.x(q)
    qc.mcx(list(atom_qubits), anc_qubit, mode="noancilla")
    for q in atom_qubits:
        qc.x(q)


def build_sqmg_hybrid_chain_circuit(
    n_atoms: int = N_ATOMS,
    atom_layers: int = 2,
    bond_layers: int = 1,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build SQMG/QCNC-style hybrid QMG for a 5-site chain with in-circuit quantum masking.

    Core architecture (PDF spirit):
    - Atom registers: 5 independent blocks × 3 qubits = 15 qubits (no reuse)
    - Bond register: 2 qubits reused across 4 chain bonds with reset/measure/reset
    - Ancillas: 2 qubits used to compute (atom_i == 000) and (atom_{i+1} == 000) reversibly

    IMPORTANT: No classical feedforward is used (no if_test/c_if). The PDF conditional bond
    module is implemented via *quantum-controlled masking*:
      apply bond_ansatz only when atom_i != 000 and atom_{i+1} != 000.

    Atom 3-bit decode mapping (fixed):
      000 -> NONE
      001 -> C
      010 -> O
      011 -> N
      100 -> S
      101 -> P
      110 -> F
      111 -> Cl

    Bond 2-bit decode mapping (fixed):
      00 -> NONE
      01 -> SINGLE
      10 -> DOUBLE
      11 -> TRIPLE

    Measurements / classical bits:
      atoms: 5×3 = 15 bits (ca0..ca4)
      bonds: 4×2 = 8 bits  (cb0..cb3)
      total: 23 bits
    """
    assert n_atoms == N_ATOMS, "This project is fixed to N=5"

    # 5 atom quantum registers (3 qubits each)
    q_atoms = [QuantumRegister(ATOM_Q, f"qa{i}") for i in range(n_atoms)]
    # bond register (2 qubits), reused
    q_bond = QuantumRegister(BOND_Q, "qb")
    # ancillas used for quantum-controlled masking
    q_anc = QuantumRegister(2, "anc")

    # 5 atom classical registers (3 bits each)
    c_atoms = [ClassicalRegister(ATOM_Q, f"ca{i}") for i in range(n_atoms)]
    # 4 bond classical registers (2 bits each) for chain edges
    c_bonds = [ClassicalRegister(BOND_Q, f"cb{i}") for i in range(len(EDGE_LIST))]

    qc = QuantumCircuit(*q_atoms, q_bond, q_anc, *c_atoms, *c_bonds)

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

    # Atom blocks (unitary). Do NOT measure yet (masking uses quantum control).
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

    # Bond blocks (dynamic + reuse)
    for bond_idx, (i, j) in enumerate(EDGE_LIST):
        qc.reset(q_bond)

        # compute none flags into ancillas
        _compute_none_flag(qc, q_atoms[i], q_anc[0])  # anc0 = 1 iff atom_i == 000
        _compute_none_flag(qc, q_atoms[j], q_anc[1])  # anc1 = 1 iff atom_j == 000

        # invert to get active flags: active=1 iff atom != 000
        qc.x(q_anc[0])
        qc.x(q_anc[1])

        # apply bond ansatz only if (active_i AND active_j)
        # NOTE: Do not use Gate.control() with unbound Parameters (fails in qiskit 0.46).
        # Instead, apply the bond ansatz with explicit 2-controlled rotations and a 3-controlled
        # entangler (controls = active_i, active_j, bond_q0).
        bp_idx = 0
        controls = [q_anc[0], q_anc[1]]
        for _ in range(bond_layers):
            for q in range(BOND_Q):
                qc.mcry(bond_params[bp_idx], controls, q_bond[q], None, mode="noancilla")
                # qiskit 0.46's mcrz/control() paths attempt to materialize a matrix and fail for
                # unbound Parameters. Implement controlled-RZ via basis change:
                #   RZ(θ) = H · RX(θ) · H
                qc.h(q_bond[q])
                qc.mcrx(bond_params[bp_idx + 1], controls, q_bond[q])
                qc.h(q_bond[q])
                bp_idx += 2
            qc.mcx([q_anc[0], q_anc[1], q_bond[0]], q_bond[1], mode="noancilla")

        # revert active -> none
        qc.x(q_anc[0])
        qc.x(q_anc[1])

        # uncompute none flags so ancillas return to |0>
        _compute_none_flag(qc, q_atoms[j], q_anc[1])
        _compute_none_flag(qc, q_atoms[i], q_anc[0])

        # measure and reset bond qubits for next edge
        qc.measure(q_bond, c_bonds[bond_idx])
        qc.reset(q_bond)

    # Final atom measurements (decode uses ca0..ca4)
    for i in range(n_atoms):
        qc.measure(q_atoms[i], c_atoms[i])

    return qc, params


__all__ = [
    "ATOM_Q",
    "BOND_Q",
    "EDGE_LIST",
    "N_ATOMS",
    "build_sqmg_hybrid_chain_circuit",
]
