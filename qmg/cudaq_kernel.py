from __future__ import annotations

from typing import List, Tuple

import importlib

# Constants are kept for reference only. Do not use them inside kernels.
N_ATOMS = 5
ATOM_Q = 3
BOND_Q = 2
ANC_Q = 2


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

    try:
        kernel, params = cudaq.make_kernel(list)
        q = kernel.qalloc(n_atoms * atom_q + bond_q + anc_q)
        bond_start = n_atoms * atom_q
        anc_start = bond_start + bond_q

        x_k, x_t = cudaq.make_kernel(cudaq.qubit)
        x_k.x(x_t)

        cx_k, cx_c, cx_t = cudaq.make_kernel(cudaq.qubit, cudaq.qubit)
        cx_k.control(x_k, cx_c, cx_t)

        ccx_k, ccx_c0, ccx_c1, ccx_t = cudaq.make_kernel(
            cudaq.qubit, cudaq.qubit, cudaq.qubit
        )
        ccx_k.control(cx_k, ccx_c0, ccx_c1, ccx_t)

        def _x(target):
            kernel.x(target)

        def _ry(theta, target):
            kernel.ry(theta, target)

        def _rz(theta, target):
            kernel.rz(theta, target)

        def _reset(target):
            kernel.reset(target)

        def _mz(target, reg_name):
            kernel.mz(target, str(reg_name))

        def _mcx(controls, target):
            if len(controls) == 1:
                kernel.control(x_k, controls[0], target)
                return
            if len(controls) == 2:
                kernel.control(cx_k, controls[0], controls[1], target)
                return
            if len(controls) == 3:
                kernel.control(ccx_k, controls[0], controls[1], controls[2], target)
                return
            raise RuntimeError(f"Unsupported control count: {len(controls)}")

        def _cry2(theta, c0, c1, target):
            kernel.cry(theta, [c0, c1], target)

        def _crz2(theta, c0, c1, target):
            kernel.crz(theta, [c0, c1], target)

        def _compute_none_flag(atom_indices, anc_idx) -> None:
            for idx in atom_indices:
                _x(q[idx])
            _mcx([q[i] for i in atom_indices], q[anc_idx])
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
                _mcx([q[base]], q[base + 1])
                _mcx([q[base + 1]], q[base + 2])

        for edge_idx, (i, j) in enumerate(edges):
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
                _mcx(
                    [q[anc_start + 0], q[anc_start + 1], q[bond_start + 0]],
                    q[bond_start + 1],
                )

            _x(q[anc_start + 0])
            _x(q[anc_start + 1])

            _compute_none_flag(atom_j, anc_start + 1)
            _compute_none_flag(atom_i, anc_start + 0)

            _mz(q[bond_start + 0], f"b{edge_idx:02d}_0")
            _mz(q[bond_start + 1], f"b{edge_idx:02d}_1")
            _reset(q[bond_start + 0])
            _reset(q[bond_start + 1])
            _reset(q[anc_start + 0])
            _reset(q[anc_start + 1])

        for atom_idx in range(n_atoms):
            for off in range(atom_q):
                _mz(q[atom_idx * atom_q + off], f"a{atom_idx:02d}_{off}")
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
