from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging
import numpy as np

try:
    import cudaq
except ImportError:  # pragma: no cover - optional dependency
    cudaq = None

from env import ATOM_VOCAB, BOND_VOCAB, EDGE_LIST, FiveAtomMolEnv
from qmg.cudaq_kernel import build_sqmg_cudaq_kernel
from qmg.generator import SampledBatch

logger = logging.getLogger(__name__)


def _set_cudaq_target(device: str) -> str:
    device = device.lower()
    if device in ("cuda-gpu", "gpu", "nvidia"):
        try:
            cudaq.set_target("nvidia")
            return "nvidia"
        except Exception:
            cudaq.set_target("qpp-cpu")
            return "qpp-cpu"
    if device in ("cuda-cpu", "cpu", "qpp", "qpp-cpu"):
        cudaq.set_target("qpp-cpu")
        return "qpp-cpu"

    # auto/unknown: default to CPU target
    try:
        cudaq.set_target("qpp-cpu")
    except Exception as exc:
        print(f"[warn] cudaq.set_target failed for qpp-cpu: {exc}")
    print(f"[warn] Unknown CUDA-Q device '{device}', defaulting to qpp-cpu.")
    return "qpp-cpu"


def _set_cudaq_seed(seed: int | None) -> None:
    if seed is None:
        return
    if hasattr(cudaq, "set_random_seed"):
        try:
            cudaq.set_random_seed(seed)
        except Exception:
            pass
    if hasattr(cudaq, "set_seed"):
        try:
            cudaq.set_seed(seed)
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
        if cudaq is None:
            raise RuntimeError("cudaq is not installed. Install requirements-cudaq.txt.")
        self.atom_layers = int(atom_layers)
        self.bond_layers = int(bond_layers)
        self.rng = np.random.default_rng(seed)
        self.env = FiveAtomMolEnv(repair_bonds=repair_bonds)
        self.device = device

        self.kernel, self.num_params = build_sqmg_cudaq_kernel(
            atom_layers=self.atom_layers, bond_layers=self.bond_layers
        )
        self.weights = self.rng.normal(0.0, 0.2, size=self.num_params)

        _set_cudaq_seed(seed)
        self.target = _set_cudaq_target(device)
        logger.debug(
            "cudaq_version=%s target=%s",
            getattr(cudaq, "__version__", "unknown"),
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

    def _expand_counts(self, counts: Dict[str, int], shots: int) -> List[str]:
        samples: List[str] = []
        for bitstring, count in counts.items():
            samples.extend([bitstring] * int(count))
        if len(samples) < shots:
            samples.extend([samples[-1]] * (shots - len(samples)))
        return samples[:shots]

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
        counts = cudaq.sample(self.kernel, self.weights.tolist(), shots_count=batch_size)
        if hasattr(counts, "counts"):
            count_map = counts.counts()
        else:
            count_map = counts
        samples = self._expand_counts(count_map, batch_size)

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
