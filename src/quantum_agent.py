from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumPolicy(nn.Module):
    """
    Quantum policy using a RealAmplitudes feature map + EstimatorQNN.
    Outputs 11 continuous values mapped to [-pi, pi]:
      - first 10 -> rotation params for environment
      - last 1  -> structure selector (later rounded to 0..2)
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        num_qubits = 4
        feature_map = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="full")
        qnn = EstimatorQNN(
            circuit=feature_map,
            input_params=feature_map.parameters,
            weight_params=[],
            observables=[nn.Identity()],  # dummy, outputs scalar
        )
        self.q_layer = TorchConnector(qnn)
        self.head = nn.Linear(1, 11)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is dummy input, shape (batch, 1); q_layer returns (batch, 1)
        q_out = self.q_layer(x)
        out = self.head(q_out)
        return torch.tanh(out) * torch.pi  # map to [-pi, pi]

    def sample_action(self) -> torch.Tensor:
        dummy = torch.zeros((1, 1), dtype=torch.float32)
        with torch.set_grad_enabled(True):
            params = self.forward(dummy)
        return params

    def update(self, log_prob: torch.Tensor, reward: float):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
