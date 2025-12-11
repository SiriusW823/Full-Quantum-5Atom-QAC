from typing import List
import torch
import math


def encode_state(history: List[int], n_wires: int = 9) -> torch.Tensor:
    """
    Map a 9-step discrete history (tokens 0-3) into rotation angles in [0, pi].
    The sequence is padded/truncated to match the qubit count for AngleEmbedding.
    """
    angles = [(math.pi / 3.0) * int(x) for x in history]  # 0..3 -> 0..pi
    if len(angles) < n_wires:
        angles += [0.0] * (n_wires - len(angles))
    else:
        angles = angles[:n_wires]
    return torch.tensor(angles, dtype=torch.float32)
