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

        def _mcx(controls, target):
            last_exc = None
            if len(controls) == 1:
                try:
                    kernel.cx(controls[0], target)
                    return
                except Exception as exc:
                    last_exc = exc
            if len(controls) == 2:
                try:
                    kernel.ccx(controls[0], controls[1], target)
                    return
                except Exception as exc:
                    last_exc = exc
            try:
                kernel.mcx(controls, target)
                return
            except Exception as exc:
                last_exc = exc
            raise RuntimeError("cudaq kernel missing multi-controlled X (mcx/ccx/cx)") from last_exc

        def _controlled_rot(
            gate: Callable[[float, object], None],
            controls: List[object],
            target: object,
            theta: float,
            name: str,
        ) -> None:
            last_exc = None
            if name == "ry":
                try:
                    kernel.mcry(theta, controls, target)
                    return
                except Exception as exc:
                    last_exc = exc
                try:
                    kernel.mcry(controls, target, theta)
                    return
                except Exception as exc:
                    last_exc = exc
            if name == "rz":
                try:
                    kernel.mcrz(theta, controls, target)
                    return
                except Exception as exc:
                    last_exc = exc
                try:
                    kernel.mcrz(controls, target, theta)
                    return
                except Exception as exc:
                    last_exc = exc
            try:
                kernel.control(controls, lambda: gate(theta, target))
                return
            except Exception as exc:
                last_exc = exc
            try:
                kernel.ctrl(controls, lambda: gate(theta, target))
                return
            except Exception as exc:
                last_exc = exc
            raise RuntimeError(f"cudaq kernel missing controlled {name} gate") from last_exc

        def _mz(qubit) -> None:
            try:
                kernel.mz(qubit)
                return
            except Exception:
                pass
            try:
                kernel.measure(qubit)
                return
            except Exception as exc:
                raise RuntimeError(
                    "No measurement op (mz/measure) on this CUDA-Q PyKernel"
                ) from exc

        def _compute_none_flag(atom_indices, anc_idx) -> None:
            for idx in atom_indices:
                kernel.x(q[idx])
            _mcx([q[i] for i in atom_indices], q[anc_idx])
            for idx in atom_indices:
                kernel.x(q[idx])

        p = 0
        for atom_idx in range(n_atoms):
            base = atom_idx * atom_q
            for _ in range(atom_layers):
                for off in range(atom_q):
                    kernel.ry(params[p], q[base + off])
                    kernel.rz(params[p + 1], q[base + off])
                    p += 2
                _mcx([q[base]], q[base + 1])
                _mcx([q[base + 1]], q[base + 2])

        for i, j in edges:
            kernel.reset(q[bond_start + 0])
            kernel.reset(q[bond_start + 1])

            atom_i = [i * atom_q + 0, i * atom_q + 1, i * atom_q + 2]
            atom_j = [j * atom_q + 0, j * atom_q + 1, j * atom_q + 2]

            _compute_none_flag(atom_i, anc_start + 0)
            _compute_none_flag(atom_j, anc_start + 1)

            kernel.x(q[anc_start + 0])
            kernel.x(q[anc_start + 1])

            bp = num_atom_params
            for _ in range(bond_layers):
                for bq in range(bond_q):
                    _controlled_rot(kernel.ry, [q[anc_start + 0], q[anc_start + 1]], q[bond_start + bq], params[bp], "ry")
                    _controlled_rot(kernel.rz, [q[anc_start + 0], q[anc_start + 1]], q[bond_start + bq], params[bp + 1], "rz")
                    bp += 2
                _mcx(
                    [q[anc_start + 0], q[anc_start + 1], q[bond_start + 0]],
                    q[bond_start + 1],
                )

            kernel.x(q[anc_start + 0])
            kernel.x(q[anc_start + 1])

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
