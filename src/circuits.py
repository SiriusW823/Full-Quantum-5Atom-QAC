import pennylane as qml


def _strong_layers_with_depth(weights, num_layers: int, n_wires: int):
    depth = max(1, int(num_layers))
    qml.StronglyEntanglingLayers(weights[:depth], wires=range(n_wires))


def actor_qnode(n_wires: int = 9, max_layers: int = 5):
    """
    Quantum policy network (Agent A) with dynamic depth.
    Uses up to 5 StronglyEntanglingLayers; actual depth is provided at call-time.
    """
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (max_layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights, num_layers: int = max_layers):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        _strong_layers_with_depth(weights, num_layers, n_wires)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit, weight_shapes


def critic_qnode(n_wires: int = 9, max_layers: int = 5):
    """
    Quantum value network (Agent B) with dynamic depth.
    Mirrors the actor architecture but outputs a single state-value scalar.
    """
    dev = qml.device("default.qubit", wires=n_wires)
    weight_shapes = {"weights": (max_layers, n_wires, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights, num_layers: int = max_layers):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        _strong_layers_with_depth(weights, num_layers, n_wires)
        return qml.expval(qml.PauliZ(0))

    return circuit, weight_shapes
