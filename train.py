"""
Train a two-layer neural network on synthetic spiral data.

Run:
    python train.py
"""

import numpy as np
from src.layers import DenseLayer
from src.network import NeuralNetwork


def make_spiral(n_samples: int = 800, n_classes: int = 3, noise: float = 0.15):
    """Spiral dataset (classic OOAD / CS toy problem)."""
    X, y = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        ix = np.arange(per_class)
        r = ix / per_class
        t = ix / per_class * 4 * np.pi + (2 * np.pi * c / n_classes)
        t += np.random.randn(per_class) * noise
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(per_class, c))
    return np.vstack(X), np.hstack(y)


def main():
    np.random.seed(42)
    X, y = make_spiral(n_samples=900, n_classes=3)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    net = NeuralNetwork(
        [
            DenseLayer(2, 64, activation="relu"),
            DenseLayer(64, 64, activation="relu"),
            DenseLayer(64, 3, activation=None),
        ],
        task="classification",
    )

    print("Training on spiral dataset (900 samples, 3 classes)...")
    net.train(X_train, y_train, epochs=40, batch_size=32, lr=5e-3, optimizer="adam")

    train_acc = net.score(X_train, y_train)
    test_acc = net.score(X_test, y_test)
    print(f"\nTrain accuracy : {train_acc:.2%}")
    print(f"Test  accuracy : {test_acc:.2%}")


if __name__ == "__main__":
    main()
