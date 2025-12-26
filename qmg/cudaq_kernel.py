from __future__ import annotations

from typing import List, Tuple, Callable

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

        mz = kernel.mz if hasattr(kernel, "mz") else kernel.measure
        reset = kernel.reset
        ry = kernel.ry
        rz = kernel.rz
        x = kernel.x

        def _mcx(controls, target):
            if len(controls) == 1 and hasattr(kernel, "cx"):
                kernel.cx(controls[0], target)
                return
            if len(controls) == 2 and hasattr(kernel, "ccx"):
                kernel.ccx(controls[0], controls[1], target)
                return
            if hasattr(kernel, "mcx"):
                kernel.mcx(controls, target)
                return
            raise RuntimeError("cudaq kernel missing multi-controlled X (mcx/ccx/cx)")

        def _controlled_rot(
            gate: Callable[[float, object], None],
            controls: List[object],
            target: object,
            theta: float,
            name: str,
        ) -> None:
            method = getattr(kernel, f"mc{name}", None) or getattr(kernel, f"mcr{name}", None)
            if method is not None:
                method(theta, controls, target)
                return
            ctrl = getattr(kernel, "control", None) or getattr(kernel, "ctrl", None)
            if ctrl is not None:
                ctrl(controls, lambda: gate(theta, target))
                return
            raise RuntimeError(f"cudaq kernel missing controlled {name} gate")

        def _compute_none_flag(atom_indices, anc_idx) -> None:
            for idx in atom_indices:
                x(q[idx])
            _mcx([q[i] for i in atom_indices], q[anc_idx])
            for idx in atom_indices:
                x(q[idx])

        p = 0
        for atom_idx in range(n_atoms):
            base = atom_idx * atom_q
            for _ in range(atom_layers):
                for off in range(atom_q):
                    ry(params[p], q[base + off])
                    rz(params[p + 1], q[base + off])
                    p += 2
                _mcx([q[base]], q[base + 1])
                _mcx([q[base + 1]], q[base + 2])

        for i, j in edges:
            reset(q[bond_start + 0])
            reset(q[bond_start + 1])

            atom_i = [i * atom_q + 0, i * atom_q + 1, i * atom_q + 2]
            atom_j = [j * atom_q + 0, j * atom_q + 1, j * atom_q + 2]

            _compute_none_flag(atom_i, anc_start + 0)
            _compute_none_flag(atom_j, anc_start + 1)

            x(q[anc_start + 0])
            x(q[anc_start + 1])

            bp = num_atom_params
            for _ in range(bond_layers):
                for bq in range(bond_q):
                    _controlled_rot(ry, [q[anc_start + 0], q[anc_start + 1]], q[bond_start + bq], params[bp], "ry")
                    _controlled_rot(rz, [q[anc_start + 0], q[anc_start + 1]], q[bond_start + bq], params[bp + 1], "rz")
                    bp += 2
                _mcx([q[anc_start + 0], q[anc_start + 1], q[bond_start + 0]], q[bond_start + 1])

            x(q[anc_start + 0])
            x(q[anc_start + 1])

            _compute_none_flag(atom_j, anc_start + 1)
            _compute_none_flag(atom_i, anc_start + 0)

            mz(q[bond_start + 0])
            mz(q[bond_start + 1])

        for idx in range(n_atoms * atom_q):
            mz(q[idx])
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
