"""Tests for activation functions."""

import numpy as np
import pytest
from src.activations import relu, relu_grad, sigmoid, sigmoid_grad, softmax, tanh, tanh_grad


class TestRelu:
    def test_positive(self):
        assert relu(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_negative(self):
        assert relu(np.array([-1.0, -5.0])).tolist() == [0.0, 0.0]

    def test_zero(self):
        assert relu(np.array([0.0])).tolist() == [0.0]

    def test_grad_positive(self):
        assert relu_grad(np.array([3.0])).tolist() == [1.0]

    def test_grad_negatize(self):
        assert relu_grad(np.array([-2.0])).tolist() == [0.0]


class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9

    def test_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] > 0.99

    def test_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] < 0.01

    def test_grad_at_zero(self):
        # grad = 0.5 * 0.5 = 0.25
        assert abs(sigmoid_grad(np.array([0.0]))[0] - 0.25) < 1e-9

    def test_output_range(self):
        z = np.linspace(-10, 10, 100)
        s = sigmoid(z)
        assert np.all(s > 0) and np.all(s < 1)


class TestSoftmax:
    def test_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0]])
        row_sums = softmax(z).sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0], atol=1e-12)

    def test_all_positive(self):
        z = np.array([[-1.0, 0.0, 1.0]])
        assert np.all(softmax(z) > 0)

    def test_argmax_preserved(self):
        z = np.array([[0.1, 5.0, 0.2]])
        assert np.argmax(softmax(z)) == 1

    def test_batch(self):
        z = np.random.randn(8, 4)
        row_sums = softmax(z).sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(8), atol=1e-12)


class TestTanh:
    def test_zero(self):
        assert abs(tanh(np.array([0.0]))[0]) < 1e-9

    def test_grad_at_zero(self):
        assert abs(tanh_grad(np.array([0.0]))[0] - 1.0) < 1e-9
