from __future__ import annotations

from typing import List, Tuple

try:
    import cudaq as _cudaq
except ImportError:  # pragma: no cover - optional dependency
    cudaq = None
    mz = reset = rx = ry = rz = x = None
else:
    cudaq = _cudaq

    def _require_gate(name: str, alternates: tuple[str, ...] = ()):
        for candidate in (name,) + alternates:
            gate = getattr(cudaq, candidate, None)
            if gate is not None:
                return gate
        raise ImportError(f"cudaq gate '{name}' not available")

    mz = _require_gate("mz", ("measure",))
    reset = _require_gate("reset")
    rx = _require_gate("rx")
    ry = _require_gate("ry")
    rz = _require_gate("rz")
    x = _require_gate("x")

# Constants are kept for reference only. Do not use them inside kernels.
N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2
ANC_Q = 2


def _compute_none_flag(q, atom_indices, anc_idx) -> None:
    """Compute ancilla = 1 iff atom code == 000 (NONE)."""
    for idx in atom_indices:
        x(q[idx])
    x.ctrl([q[i] for i in atom_indices], q[anc_idx])
    for idx in atom_indices:
        x(q[idx])


def build_sqmg_cudaq_kernel(
    atom_layers: int = 2,
    bond_layers: int = 1,
) -> Tuple["cudaq.Kernel", int]:
    if cudaq is None:
        raise RuntimeError("cudaq is not available")

    n_atoms = 5
    atom_q = 3
    bond_q = 2
    anc_q = 2
    edges = (
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
    )

    num_atom_params = n_atoms * atom_layers * atom_q * 2
    num_bond_params = bond_layers * bond_q * 2
    num_params = num_atom_params + num_bond_params

    @cudaq.kernel
    def kernel(params: List[float]):
        q = cudaq.qvector(n_atoms * atom_q + bond_q + anc_q)
        bond_start = n_atoms * atom_q
        anc_start = bond_start + bond_q

        p = 0
        for atom_idx in range(n_atoms):
            base = atom_idx * atom_q
            for _ in range(atom_layers):
                for off in range(atom_q):
                    ry(params[p], q[base + off])
                    rz(params[p + 1], q[base + off])
                    p += 2
                x.ctrl(q[base], q[base + 1])
                x.ctrl(q[base + 1], q[base + 2])

        for i, j in edges:
            # reset bond qubits
            reset(q[bond_start + 0])
            reset(q[bond_start + 1])

            atom_i = [i * atom_q + 0, i * atom_q + 1, i * atom_q + 2]
            atom_j = [j * atom_q + 0, j * atom_q + 1, j * atom_q + 2]

            _compute_none_flag(q, atom_i, anc_start + 0)
            _compute_none_flag(q, atom_j, anc_start + 1)

            # active flags
            x(q[anc_start + 0])
            x(q[anc_start + 1])

            bp = num_atom_params
            for _ in range(bond_layers):
                for bq in range(bond_q):
                    ry.ctrl(
                        [q[anc_start + 0], q[anc_start + 1]],
                        q[bond_start + bq],
                        params[bp],
                    )
                    rz.ctrl(
                        [q[anc_start + 0], q[anc_start + 1]],
                        q[bond_start + bq],
                        params[bp + 1],
                    )
                    bp += 2
                x.ctrl(
                    [q[anc_start + 0], q[anc_start + 1], q[bond_start + 0]],
                    q[bond_start + 1],
                )

            x(q[anc_start + 0])
            x(q[anc_start + 1])

            _compute_none_flag(q, atom_j, anc_start + 1)
            _compute_none_flag(q, atom_i, anc_start + 0)

            # measure bond qubits (2 bits per edge)
            mz(q[bond_start + 0])
            mz(q[bond_start + 1])

        # measure atoms (15 bits)
        for idx in range(n_atoms * atom_q):
            mz(q[idx])

    return kernel, num_params


__all__ = [
    "build_sqmg_cudaq_kernel",
    "N_ATOMS",
    "ATOM_Q",
    "BOND_Q",
    "ANC_Q",
]
