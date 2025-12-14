import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumPolicy(nn.Module):
    """
    Quantum actor-critic using EstimatorQNN backbone.
    - Actor head: 11 continuous outputs mapped to [-pi, pi] (10 angles + 1 structure selector).
    - Critic head: scalar V(s) baseline.
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        num_qubits = 4
        feature_map = ZFeatureMap(feature_dimension=1)  # input dim = 1
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="full")
        qc = QuantumCircuit(num_qubits)
        qc.append(feature_map, [0])  # apply 1-qubit feature map to qubit 0
        qc.append(ansatz, range(num_qubits))  # apply ansatz over all qubits
        self.num_inputs = 1
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        self.q_layer = TorchConnector(qnn)
        self.actor_head = nn.Linear(1, 11)
        self.value_head = nn.Linear(1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        # x is dummy input, shape (batch, 1); q_layer returns (batch, 1)
        if x.ndim == 1:
            x = x.view(1, -1)
        q_out = self.q_layer(x)
        actor_out = torch.tanh(self.actor_head(q_out)) * torch.pi  # map to [-pi, pi]
        value_out = self.value_head(q_out)  # scalar baseline
        return actor_out, value_out

    def sample_action(self) -> torch.Tensor:
        dummy = torch.zeros((1, self.num_inputs), dtype=torch.float32)
        with torch.set_grad_enabled(True):
            actor_out, value_out = self.forward(dummy)
        params = actor_out
        if self.training:
            EXPLORATION_NOISE_STD = 0.5
            noise = torch.randn_like(params) * EXPLORATION_NOISE_STD
            params = params + noise
        return params, value_out

    def update(self, actor_out: torch.Tensor, value_out: torch.Tensor, reward: float):
        # Advantage with baseline
        advantage = reward - value_out
        value_loss = 0.5 * advantage.pow(2).mean()
        policy_loss = -(actor_out.mean() * advantage.detach())  # proxy for log pi
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
