"""NeuralNetwork: stacks DenseLayers, runs training loop."""

import numpy as np
from .layers import DenseLayer
from .loss import cross_entropy_loss, cross_entropy_grad, mse_loss, mse_grad
from .optimizer import SGD, Adam
from .activations import softmax


class NeuralNetwork:
    """
    Flexible multi-layer neural network for classification or regression.

    Example
    -------
    >>> net = NeuralNetwork([
    ...     DenseLayer(784, 128, activation='relu'),
    ...     DenseLayer(128, 64,  activation='relu'),
    ...     DenseLayer(64,  10,  activation=None),
    ... ], task='classification')
    >>> net.train(X_train, y_train, epochs=20, batch_size=64, lr=1e-3)
    """

    def __init__(self, layers: list[DenseLayer], task: str = "classification"):
        self.layers = layers
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task

    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        if self.task == "classification":
            out = softmax(out)
        return out

    def _backward(self, probs: np.ndarray, y: np.ndarray) -> None:
        if self.task == "classification":
            grad = cross_entropy_grad(probs, y)
        else:
            grad = mse_grad(probs, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class indices (classification) or raw outputs (regression)."""
        out = self._forward(x)
        if self.task == "classification":
            return np.argmax(out, axis=1)
        return out

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for classification; R^2 for regression (not implemented here)."""
        preds = self.predict(x)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        optimizer: str = "adam",
        verbose: bool = True,
    ) -> list[float]:
        """
        Mini-batch training loop.

        Returns list of per-epoch loss values.
        """
        n = x.shape[0]
        opt = Adam(lr=lr) if optimizer == "adam" else SGD(lr=lr, momentum=0.9)
        history = []

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            x_s, y_s = x[idx], y[idx]
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                xb = x_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]

                probs = self._forward(xb)
                if self.task == "classification":
                    loss = cross_entropy_loss(probs, yb)
                else:
                    loss = mse_loss(probs, yb)
                epoch_loss += loss
                batches += 1

                self._backward(probs, yb)
                opt.step(self.layers)

            avg_loss = epoch_loss / batches
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

        return history
