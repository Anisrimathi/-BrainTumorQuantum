
import pennylane as qml
import torch
import torch.nn as nn


# Quantum circuit configuration
N_QUBITS = 8
N_LAYERS = 2

STRATEGIES = ("linear", "ring", "fully_connected")


def make_entanglement_edges(strategy):
    """Build the entanglement connections for each circuit topology."""

    if strategy == "linear":
        return [(i, i + 1) for i in range(N_QUBITS - 1)]

    if strategy == "ring":
        return (
            [(i, i + 1) for i in range(N_QUBITS - 1)]
            + [(N_QUBITS - 1, 0)]
        )

    if strategy == "fully_connected":
        return [
            (i, j)
            for i in range(N_QUBITS)
            for j in range(i + 1, N_QUBITS)
        ]

    raise ValueError(
        "Invalid strategy. Choose linear, ring, or fully_connected."
    )


class QuantumVQC(nn.Module):
    """
    Eight-qubit variational quantum circuit.

    Input:
        [8] or [batch_size, 8]

    Output:
        [8] or [batch_size, 8]

    The output contains Pauli-Z expectation values.
    """

    def __init__(self, strategy="ring", n_layers=N_LAYERS):
        super().__init__()

        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose one of {STRATEGIES}."
            )

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.strategy = strategy
        self.n_layers = n_layers
        self.edges = make_entanglement_edges(strategy)

        # CPU-based state-vector quantum simulator.
        self.device_name = "default.qubit"
        quantum_device = qml.device(
            self.device_name,
            wires=N_QUBITS,
        )

        @qml.qnode(
            quantum_device,
            interface="torch",
            diff_method="backprop",
        )
        def circuit(inputs, weights):
            # Encode the eight classical features.
            for wire in range(N_QUBITS):
                qml.RY(inputs[wire], wires=wire)

            # Variational rotations and entanglement.
            for layer in range(self.n_layers):
                for wire in range(N_QUBITS):
                    qml.Rot(
                        weights[layer, wire, 0],
                        weights[layer, wire, 1],
                        weights[layer, wire, 2],
                        wires=wire,
                    )

                for control, target in self.edges:
                    qml.CNOT(wires=[control, target])

            return [
                qml.expval(qml.PauliZ(wire))
                for wire in range(N_QUBITS)
            ]

        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes={
                "weights": (self.n_layers, N_QUBITS, 3)
            },
        )

    def forward(self, x):
        if x.ndim == 1:
            if x.shape[0] != N_QUBITS:
                raise ValueError(
                    f"Expected {N_QUBITS} features, got {x.shape[0]}."
                )
            return self.quantum_layer(x)

        if x.ndim == 2:
            if x.shape[1] != N_QUBITS:
                raise ValueError(
                    f"Expected input shape [batch, {N_QUBITS}], "
                    f"got {tuple(x.shape)}."
                )

            # Process samples separately to avoid QNode batch-shape
            # ambiguity. This is compatible but may be slower.
            outputs = [
                self.quantum_layer(sample)
                for sample in x.unbind(dim=0)
            ]
            return torch.stack(outputs, dim=0)

        raise ValueError(
            "Input must have shape [8] or [batch_size, 8]. "
            f"Received {tuple(x.shape)}."
        )


def test_quantum_circuit():
    """Test all topologies, output shapes, and gradient flow."""

    torch.manual_seed(42)

    print("PennyLane version:", qml.__version__)
    print("PyTorch version:", torch.__version__)
    print("Qubits:", N_QUBITS)
    print("Variational layers:", N_LAYERS)
    print("Simulator: default.qubit (CPU)")
    print("-" * 55)

    for strategy in STRATEGIES:
        print(f"\nTesting strategy: {strategy}")

        model = QuantumVQC(
            strategy=strategy,
            n_layers=N_LAYERS,
        )

        # Two samples, each containing eight features.
        inputs = torch.randn(
            2,
            N_QUBITS,
            dtype=torch.float32,
            requires_grad=True,
        )

        outputs = model(inputs)

        print("Input shape:", tuple(inputs.shape))
        print("Output shape:", tuple(outputs.shape))
        print(
            "Output finite:",
            bool(torch.isfinite(outputs).all()),
        )
        print(
            "Output range:",
            (
                round(outputs.min().item(), 4),
                round(outputs.max().item(), 4),
            ),
        )

        loss = outputs.square().mean()
        loss.backward()

        print("Loss:", round(loss.item(), 6))
        print(
            "Input gradients available:",
            inputs.grad is not None
            and bool(torch.isfinite(inputs.grad).all()),
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        print("Trainable circuit parameters:", parameter_count)

        assert outputs.shape == (2, N_QUBITS)
        assert bool(torch.isfinite(outputs).all())
        assert inputs.grad is not None
        assert bool(torch.isfinite(inputs.grad).all())

    print("\nAll quantum circuit tests passed.")


if __name__ == "__main__":
    test_quantum_circuit()
