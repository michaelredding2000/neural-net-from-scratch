"""End-to-end network tests."""

import numpy as np
import pytest
from src.layers import DenseLayer
from src.network import NeuralNetwork


def make_xor(n=200):
    """XOR-style synthetic data -- linearly inseparable."""
    np.random.seed(7)
    X = np.random.randn(n, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


class TestNetworkPredict:
    def test_output_shape(self):
        net = NeuralNetwork([
            DenseLayer(2, 8, activation="relu"),
            DenseLayer(8, 3, activation=None),
        ], task="classification")
        X = np.random.randn(10, 2)
        preds = net.predict(X)
        assert preds.shape == (10,)

    def test_predictions_in_class_range(self):
        net = NeuralNetwork([
            DenseLayer(2, 16, activation="relu"),
            DenseLayer(16, 4, activation=None),
        ], task="classification")
        X = np.random.randn(20, 2)
        preds = net.predict(X)
        assert np.all(preds >= 0) and np.all(preds < 4)


class TestNetworkTraining:
    def test_loss_decreases(self):
        """Loss after 30 epochs should be lower than initial loss."""
        np.random.seed(42)
        X, y = make_xor(n=200)
        net = NeuralNetwork([
            DenseLayer(2, 32, activation="relu"),
            DenseLayer(32, 2, activation=None),
        ], task="classification")
        history = net.train(X, y, epochs=30, batch_size=32, lr=1e-2, verbose=False)
        assert history[-1] < history[0], "Loss did not decrease after training"

    def test_accuracy_above_chance(self):
        """Network should learn XOR well above 50 % chance."""
        np.random.seed(0)
        X, y = make_xor(n=400)
        net = NeuralNetwork([
            DenseLayer(2, 64, activation="relu"),
            DenseLayer(64, 2, activation=None),
        ], task="classification")
        net.train(X, y, epochs=60, batch_size=32, lr=5e-3, verbose=False)
        acc = net.score(X, y)
        assert acc > 0.80, f"Expected >80% accuracy, got {acc:.2%}"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            NeuralNetwork([], task="unsupported")
