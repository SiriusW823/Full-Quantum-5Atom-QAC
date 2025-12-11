import pennylane as qml


def actor_qnode(n_wires: int = 9, layers: int = 4):
    """
    Quantum policy network (Agent A).
    Uses 9 qubits and StronglyEntanglingLayers to output 4 expectation values
    that become action logits via softmax on the classical side.
    """
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit, weight_shapes


def critic_qnode(n_wires: int = 9, layers: int = 4):
    """
    Quantum value network (Agent B).
    Mirrors the actor architecture but outputs a single state-value scalar.
    """
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    return circuit, weight_shapes
