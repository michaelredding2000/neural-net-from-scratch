"""Activation functions with forward and backward (gradient) implementations."""

import numpy as np


def relu(z):
    """Rectified Linear Unit."""
    return np.maximum(0, z)


def relu_grad(z):
    """Derivative of ReLU with respect to z."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def sigmoid_grad(z):
    """Derivative of sigmoid with respect to z."""
    s = sigmoid(z)
    return s * (1 - s)


def softmax(z):
    """Numerically stable softmax (row-wise for batches)."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}
