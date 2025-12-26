from __future__ import annotations

from typing import List, Tuple

try:
    import cudaq
except Exception:  # pragma: no cover - optional dependency
    cudaq = None

from env import EDGE_LIST

N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2
ANC_Q = 2


def _compute_none_flag(q, atom_indices, anc_idx) -> None:
    """Compute ancilla = 1 iff atom code == 000 (NONE)."""
    for idx in atom_indices:
        cudaq.x(q[idx])
    cudaq.x.ctrl([q[i] for i in atom_indices], q[anc_idx])
    for idx in atom_indices:
        cudaq.x(q[idx])


def build_sqmg_cudaq_kernel(
    atom_layers: int = 2,
    bond_layers: int = 1,
) -> Tuple["cudaq.Kernel", int]:
    if cudaq is None:
        raise RuntimeError("cudaq is not available")

    num_atom_params = N_ATOMS * atom_layers * ATOM_Q * 2
    num_bond_params = bond_layers * BOND_Q * 2
    num_params = num_atom_params + num_bond_params

    @cudaq.kernel
    def kernel(params: List[float]):
        q = cudaq.qvector(N_ATOMS * ATOM_Q + BOND_Q + ANC_Q)
        bond_start = N_ATOMS * ATOM_Q
        anc_start = bond_start + BOND_Q

        p = 0
        for atom_idx in range(N_ATOMS):
            base = atom_idx * ATOM_Q
            for _ in range(atom_layers):
                for off in range(ATOM_Q):
                    cudaq.ry(params[p], q[base + off])
                    cudaq.rz(params[p + 1], q[base + off])
                    p += 2
                cudaq.x.ctrl(q[base], q[base + 1])
                cudaq.x.ctrl(q[base + 1], q[base + 2])

        for edge_idx, (i, j) in enumerate(EDGE_LIST):
            # reset bond qubits
            cudaq.reset(q[bond_start + 0])
            cudaq.reset(q[bond_start + 1])

            atom_i = [i * ATOM_Q + 0, i * ATOM_Q + 1, i * ATOM_Q + 2]
            atom_j = [j * ATOM_Q + 0, j * ATOM_Q + 1, j * ATOM_Q + 2]

            _compute_none_flag(q, atom_i, anc_start + 0)
            _compute_none_flag(q, atom_j, anc_start + 1)

            # active flags
            cudaq.x(q[anc_start + 0])
            cudaq.x(q[anc_start + 1])

            bp = num_atom_params
            for _ in range(bond_layers):
                for bq in range(BOND_Q):
                    cudaq.ry.ctrl(
                        [q[anc_start + 0], q[anc_start + 1]],
                        q[bond_start + bq],
                        params[bp],
                    )
                    cudaq.rz.ctrl(
                        [q[anc_start + 0], q[anc_start + 1]],
                        q[bond_start + bq],
                        params[bp + 1],
                    )
                    bp += 2
                cudaq.x.ctrl(
                    [q[anc_start + 0], q[anc_start + 1], q[bond_start + 0]],
                    q[bond_start + 1],
                )

            cudaq.x(q[anc_start + 0])
            cudaq.x(q[anc_start + 1])

            _compute_none_flag(q, atom_j, anc_start + 1)
            _compute_none_flag(q, atom_i, anc_start + 0)

            # measure bond qubits (2 bits per edge)
            cudaq.measure(q[bond_start + 0])
            cudaq.measure(q[bond_start + 1])

        # measure atoms (15 bits)
        for idx in range(N_ATOMS * ATOM_Q):
            cudaq.measure(q[idx])

    return kernel, num_params


__all__ = [
    "build_sqmg_cudaq_kernel",
    "N_ATOMS",
    "ATOM_Q",
    "BOND_Q",
    "ANC_Q",
]
