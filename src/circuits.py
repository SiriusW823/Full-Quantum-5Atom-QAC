import pennylane as qml


def actor_qnode(n_wires: int = 9):
    """
    Quantum policy network (Agent A). Depth is controlled by the length of the
    provided weight tensor (weights.shape[0]).
    """
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(4)])

    return circuit


def critic_qnode(n_wires: int = 9):
    """
    Quantum value network (Agent B). Depth is controlled by the length of the
    provided weight tensor (weights.shape[0]).
    """
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    return circuit
