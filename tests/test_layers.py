"""Tests for DenseLayer forward and backward passes."""

import numpy as np
import pytest
from src.layers import DenseLayer


class TestDenseLayerForward:
    def test_output_shape(self):
        layer = DenseLayer(4, 8, activation="relu")
        x = np.random.randn(16, 4)
        out = layer.forward(x)
        assert out.shape == (16, 8)

    def test_relu_no_negatives(self):
        layer = DenseLayer(4, 8, activation="relu")
        x = np.random.randn(10, 4)
        out = layer.forward(x)
        assert np.all(out >= 0)

    def test_linear_activation(self):
        layer = DenseLayer(3, 5, activation=None)
        x = np.ones((2, 3))
        out = layer.forward(x)
        assert out.shape == (2, 5)

    def test_sigmoid_range(self):
        layer = DenseLayer(3, 6, activation="sigmoid")
        x = np.random.randn(5, 3)
        out = layer.forward(x)
        assert np.all(out > 0) and np.all(out < 1)


class TestDenseLayerBackward:
    def test_grad_shapes(self):
        layer = DenseLayer(4, 8, activation="relu")
        x = np.random.randn(16, 4)
        layer.forward(x)
        d_out = np.random.randn(16, 8)
        d_in = layer.backward(d_out)
        assert d_in.shape == (16, 4)
        assert layer.dW.shape == (4, 8)
        assert layer.db.shape == (1, 8)

    def test_numerical_gradient(self):
        """Finite-difference check on W gradient."""
        np.random.seed(0)
        layer = DenseLayer(3, 4, activation="relu")
        x = np.random.randn(8, 3)
        out = layer.forward(x)
        d_out = np.ones_like(out)
        layer.backward(d_out)
        analytic_dW = layer.dW.copy()

        eps = 1e-5
        numeric_dW = np.zeros_like(layer.W)
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                W_plus = layer.W.copy()
                W_minus = layer.W.copy()
                W_plus[i, j] += eps
                W_minus[i, j] -= eps

                layer.W = W_plus
                f_plus = np.sum(layer.forward(x))
                layer.W = W_minus
                f_minus = np.sum(layer.forward(x))
                numeric_dW[i, j] = (f_plus - f_minus) / (2 * eps)

        layer.W = analytic_dW  # restore (shape reused for clarity)
        np.testing.assert_allclose(analytic_dW, numeric_dW, atol=1e-4)
