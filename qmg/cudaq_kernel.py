from __future__ import annotations

from typing import List, Tuple

import importlib

# Constants are kept for reference only. Do not use them inside kernels.
N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2
ANC_Q = 3


def _import_cudaq():  # pragma: no cover - thin wrapper
    try:
        return importlib.import_module("cudaq")
    except ImportError:
        return None


cudaq = _import_cudaq()


def build_sqmg_cudaq_kernel(
    atom_layers: int = 2,
    bond_layers: int = 1,
) -> Tuple["cudaq.Kernel", int]:
    global cudaq
    if cudaq is None:
        cudaq = _import_cudaq()
    if cudaq is None:
        raise RuntimeError("cudaq is not available")
    n_atoms = 5
    atom_q = 3
    bond_q = 2
    anc_q = 3
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

    try:
        kernel, params = cudaq.make_kernel(list)
        q = kernel.qalloc(n_atoms * atom_q + bond_q + anc_q)
        bond_start = n_atoms * atom_q
        anc_start = bond_start + bond_q
        scratch = anc_start + 2

        def _x(target):
            kernel.x(target=target)

        def _ry(theta, target):
            kernel.ry(parameter=theta, target=target)

        def _rz(theta, target):
            kernel.rz(parameter=theta, target=target)

        def _reset(target):
            kernel.reset(target=target)

        def _mz(target):
            kernel.mz(target=target)

        def _cx(control, target):
            kernel.cx(control=control, target=target)

        def _ccx(c0, c1, target):
            kernel.ccx(control0=c0, control1=c1, target=target)

        def _mcx_3(c0, c1, c2, target, scratch_q):
            _reset(scratch_q)
            _ccx(c0, c1, scratch_q)
            _ccx(scratch_q, c2, target)
            _ccx(c0, c1, scratch_q)
            _reset(scratch_q)

        def _cry2(theta, c0, c1, target):
            kernel.cry(parameter=theta, controls=[c0, c1], target=target)

        def _crz2(theta, c0, c1, target):
            kernel.crz(parameter=theta, controls=[c0, c1], target=target)

        def _compute_none_flag(atom_indices, anc_idx) -> None:
            for idx in atom_indices:
                _x(q[idx])
            _mcx_3(q[atom_indices[0]], q[atom_indices[1]], q[atom_indices[2]], q[anc_idx], q[scratch])
            for idx in atom_indices:
                _x(q[idx])

        p = 0
        for atom_idx in range(n_atoms):
            base = atom_idx * atom_q
            for _ in range(atom_layers):
                for off in range(atom_q):
                    _ry(params[p], q[base + off])
                    _rz(params[p + 1], q[base + off])
                    p += 2
                _cx(q[base], q[base + 1])
                _cx(q[base + 1], q[base + 2])

        for i, j in edges:
            _reset(q[bond_start + 0])
            _reset(q[bond_start + 1])

            atom_i = [i * atom_q + 0, i * atom_q + 1, i * atom_q + 2]
            atom_j = [j * atom_q + 0, j * atom_q + 1, j * atom_q + 2]

            _compute_none_flag(atom_i, anc_start + 0)
            _compute_none_flag(atom_j, anc_start + 1)

            _x(q[anc_start + 0])
            _x(q[anc_start + 1])

            bp = num_atom_params
            for _ in range(bond_layers):
                for bq in range(bond_q):
                    _cry2(params[bp], q[anc_start + 0], q[anc_start + 1], q[bond_start + bq])
                    _crz2(params[bp + 1], q[anc_start + 0], q[anc_start + 1], q[bond_start + bq])
                    bp += 2
                _mcx_3(
                    q[anc_start + 0],
                    q[anc_start + 1],
                    q[bond_start + 0],
                    q[bond_start + 1],
                    q[scratch],
                )

            _x(q[anc_start + 0])
            _x(q[anc_start + 1])

            _compute_none_flag(atom_j, anc_start + 1)
            _compute_none_flag(atom_i, anc_start + 0)

            _mz(q[bond_start + 0])
            _mz(q[bond_start + 1])

        for idx in range(n_atoms * atom_q):
            _mz(q[idx])
    except Exception as exc:
        raise RuntimeError("cudaq is installed but failed to build SQMG kernel") from exc

    return kernel, num_params


__all__ = [
    "build_sqmg_cudaq_kernel",
    "N_ATOMS",
    "ATOM_Q",
    "BOND_Q",
    "ANC_Q",
]
