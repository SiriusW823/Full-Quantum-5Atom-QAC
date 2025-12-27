from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import importlib
import logging
import numpy as np

from env import ATOM_VOCAB, BOND_VOCAB, EDGE_LIST, FiveAtomMolEnv
from qmg.cudaq_kernel import build_sqmg_cudaq_kernel
from qmg.generator import SampledBatch

logger = logging.getLogger(__name__)


def _import_cudaq():  # pragma: no cover - optional dependency
    try:
        return importlib.import_module("cudaq")
    except ImportError:
        return None


def _set_cudaq_target(cudaq_mod, device: str) -> str:
    device = device.lower()
    if device in ("cuda-gpu", "gpu", "nvidia"):
        try:
            cudaq_mod.set_target("nvidia")
            return "nvidia"
        except Exception:
            cudaq_mod.set_target("qpp-cpu")
            return "qpp-cpu"
    if device in ("cuda-cpu", "cpu", "qpp", "qpp-cpu"):
        cudaq_mod.set_target("qpp-cpu")
        return "qpp-cpu"

    # auto/unknown: default to CPU target
    try:
        cudaq_mod.set_target("qpp-cpu")
    except Exception as exc:
        print(f"[warn] cudaq.set_target failed for qpp-cpu: {exc}")
    print(f"[warn] Unknown CUDA-Q device '{device}', defaulting to qpp-cpu.")
    return "qpp-cpu"


def _set_cudaq_seed(cudaq_mod, seed: int | None) -> None:
    if seed is None:
        return
    if hasattr(cudaq_mod, "set_random_seed"):
        try:
            cudaq_mod.set_random_seed(seed)
        except Exception:
            pass
    if hasattr(cudaq_mod, "set_seed"):
        try:
            cudaq_mod.set_seed(seed)
        except Exception:
            pass


class CudaQMGGenerator:
    """CUDA-Q SQMG generator (3N+2 qubits) with dynamic bond reuse."""

    def __init__(
        self,
        atom_layers: int = 2,
        bond_layers: int = 1,
        repair_bonds: bool = False,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        cudaq_mod = _import_cudaq()
        if cudaq_mod is None:
            raise RuntimeError("cudaq is not installed. Install requirements-cudaq.txt.")
        self._cudaq = cudaq_mod
        self.atom_layers = int(atom_layers)
        self.bond_layers = int(bond_layers)
        self.rng = np.random.default_rng(seed)
        self.env = FiveAtomMolEnv(repair_bonds=repair_bonds)
        self.device = device
        self._last_bitstrings: List[str] = []

        self.kernel, self.num_params = build_sqmg_cudaq_kernel(
            atom_layers=self.atom_layers, bond_layers=self.bond_layers
        )
        self.weights = self.rng.normal(0.0, 0.2, size=self.num_params)

        _set_cudaq_seed(self._cudaq, seed)
        self.target = _set_cudaq_target(self._cudaq, device)
        logger.debug(
            "cudaq_version=%s target=%s",
            getattr(self._cudaq, "__version__", "unknown"),
            self.target,
        )

    @property
    def num_weights(self) -> int:
        return len(self.weights)

    def get_weights(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def set_weights(self, new_w: np.ndarray) -> None:
        assert new_w.shape == self.weights.shape
        self.weights = np.array(new_w, copy=True)

    def _bond_reg_names(self) -> List[str]:
        return [
            f"b{edge_idx:02d}_{bit}"
            for edge_idx in range(len(EDGE_LIST))
            for bit in range(2)
        ]

    def _atom_reg_names(self) -> List[str]:
        return [f"a{atom_idx:02d}_{bit}" for atom_idx in range(5) for bit in range(3)]

    def _get_register_counts(self, result, name: str) -> Dict[str, int]:
        try:
            sub = result.get_register_counts(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read register counts: name={name}, result_type={type(result)}"
            ) from exc

        try:
            return {k: int(v) for k, v in sub.items()}
        except Exception:
            pass
        try:
            dct = dict(sub)
            return {k: int(v) for k, v in dct.items()}
        except Exception:
            pass
        try:
            keys = list(sub)
            out: Dict[str, int] = {}
            for k in keys:
                out[k] = int(sub[k])
            return out
        except Exception as exc:
            raise RuntimeError(
                "Failed to read register counts: "
                f"name={name}, result_type={type(result)}, sub_type={type(sub)}"
            ) from exc

    def _expand_counts_1bit(self, counts: Dict[str, int], shots: int) -> List[str]:
        samples: List[str] = []
        for bitstring, count in counts.items():
            bits = str(bitstring).strip().replace(" ", "")
            if bits.startswith("0b"):
                bits = bits[2:]
            bit = bits[-1] if bits else "0"
            samples.extend([bit] * int(count))
        if not samples:
            return ["0"] * shots
        if len(samples) < shots:
            samples.extend(["0"] * (shots - len(samples)))
        return samples[:shots]

    def _reconstruct_bitstrings(self, result, shots: int) -> List[str]:
        bond_names = self._bond_reg_names()
        atom_names = self._atom_reg_names()
        reg_order = bond_names + atom_names
        reg_bits: Dict[str, List[str]] = {}
        for name in reg_order:
            counts_map = self._get_register_counts(result, name)
            reg_bits[name] = self._expand_counts_1bit(counts_map, shots)

        samples: List[str] = []
        for shot_idx in range(shots):
            bits = "".join(reg_bits[name][shot_idx] for name in reg_order)
            samples.append(bits)
        return samples

    def _decode_shot(self, bitstring: str) -> Tuple[List[int], List[int]]:
        bits = bitstring.replace(" ", "")
        num_bond_bits = len(EDGE_LIST) * 2
        num_atom_bits = 5 * 3
        expected = num_bond_bits + num_atom_bits
        if len(bits) != expected:
            raise ValueError(
                f"Unexpected CUDA-Q bitstring length {len(bits)} != {expected}"
            )
        bond_bits = bits[:num_bond_bits]
        atom_bits = bits[num_bond_bits:]

        bond_ids = []
        for i in range(0, num_bond_bits, 2):
            code = int(bond_bits[i : i + 2], 2)
            bond_ids.append(int(code % len(BOND_VOCAB)))

        atom_ids = []
        for i in range(0, num_atom_bits, 3):
            code = int(atom_bits[i : i + 3], 2)
            atom_ids.append(int(code % len(ATOM_VOCAB)))

        return atom_ids, bond_ids

    def sample_actions(self, batch_size: int = 1) -> SampledBatch:
        result = self._cudaq.sample(
            self.kernel, self.weights.tolist(), shots_count=batch_size
        )
        samples = self._reconstruct_bitstrings(result, batch_size)
        self._last_bitstrings = samples

        atoms_batch = np.zeros((batch_size, 5), dtype=int)
        bonds_batch = np.zeros((batch_size, len(EDGE_LIST)), dtype=int)
        smiles: List[Optional[str]] = []
        valids: List[bool] = []
        uniques: List[bool] = []

        for i, bitstring in enumerate(samples):
            atom_ids, bond_ids = self._decode_shot(bitstring)
            atoms_batch[i] = atom_ids
            bonds_batch[i] = bond_ids
            before = len(self.env.seen_smiles)
            smi, v = self.env.build_smiles_from_actions(atom_ids, bond_ids)
            after = len(self.env.seen_smiles)
            smiles.append(smi)
            valids.append(v)
            uniques.append(bool(v and after > before))

        return SampledBatch(
            atoms=atoms_batch,
            bonds=bonds_batch,
            smiles=smiles,
            valids=valids,
            uniques=uniques,
        )


__all__ = ["CudaQMGGenerator"]
