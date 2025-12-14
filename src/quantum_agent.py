from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import ZFeatureMap
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
        feature_map = ZFeatureMap(feature_dimension=1)  # input dim = 1
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="full")
        qc = feature_map.compose(ansatz, front=True)
        self.num_inputs = 1
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        self.q_layer = TorchConnector(qnn)
        self.head = nn.Linear(1, 11)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is dummy input, shape (batch, 1); q_layer returns (batch, 1)
        if x.ndim == 1:
            x = x.view(1, -1)
        q_out = self.q_layer(x)
        out = self.head(q_out)
        return torch.tanh(out) * torch.pi  # map to [-pi, pi]

    def sample_action(self) -> torch.Tensor:
        dummy = torch.zeros((1, self.num_inputs), dtype=torch.float32)
        with torch.set_grad_enabled(True):
            params = self.forward(dummy)
        return params

    def update(self, log_prob: torch.Tensor, reward: float):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
