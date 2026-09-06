"""Dense (fully-connected) layer with forward and backward passes."""

import numpy as np
from .activations import ACTIVATIONS


class DenseLayer:
    """
    A single fully-connected layer.

    Parameters
    ----------
    input_size  : number of input features
    output_size : number of neurons
    activation  : 'relu' | 'sigmoid' | 'tanh' | None (linear)
    """

    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        self.activation_name = activation
        self.activation_fn, self.activation_grad = (
            ACTIVATIONS[activation] if activation else (lambda z: z, lambda z: np.ones_like(z))
        )

        # He initialisation for ReLU; Xavier otherwise
        if activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros((1, output_size))

        # Gradient accumulators (set during backward)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # Cache for backprop
        self._x: np.ndarray | None = None   # layer input
        self._z: np.ndarray | None = None   # pre-activation

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (batch, input_size)  ->  output shape: (batch, output_size)"""
        self._x = x
        self._z = x @ self.W + self.b
        return self.activation_fn(self._z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        d_out : gradient of loss w.r.t. this layer's output, shape (batch, output_size)
        Returns gradient of loss w.r.t. this layer's input, shape (batch, input_size)
        """
        d_z = d_out * self.activation_grad(self._z)    # element-wise
        self.dW = self._x.T @ d_z
        self.db = d_z.sum(axis=0, keepdims=True)
        return d_z @ self.W.T
