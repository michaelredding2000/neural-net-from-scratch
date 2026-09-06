"""Gradient-descent optimizers."""

import numpy as np
from .layers import DenseLayer


class SGD:
    """Vanilla stochastic gradient descent with optional momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict = {}   # layer_id -> (vW, vb)

    def step(self, layers: list[DenseLayer]) -> None:
        for i, layer in enumerate(layers):
            if layer.dW is None:
                continue
            if i not in self._velocity:
                self._velocity[i] = (
                    np.zeros_like(layer.W),
                    np.zeros_like(layer.b),
                )
            vW, vb = self._velocity[i]
            vW = self.momentum * vW - self.lr * layer.dW
            vb = self.momentum * vb - self.lr * layer.db
            self._velocity[i] = (vW, vb)
            layer.W += vW
            layer.b += vb


class Adam:
    """Adam optimiser (Kingma & Ba, 2014)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0
        self._m: dict = {}
        self._v: dict = {}

    def step(self, layers: list[DenseLayer]) -> None:
        self._t += 1
        for i, layer in enumerate(layers):
            if layer.dW is None:
                continue
            if i not in self._m:
                self._m[i] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
                self._v[i] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            for j, (grad, param) in enumerate(
                [(layer.dW, layer.W), (layer.db, layer.b)]
            ):
                self._m[i][j] = self.beta1 * self._m[i][j] + (1 - self.beta1) * grad
                self._v[i][j] = self.beta2 * self._v[i][j] + (1 - self.beta2) * grad ** 2
                m_hat = self._m[i][j] / (1 - self.beta1 ** self._t)
                v_hat = self._v[i][j] / (1 - self.beta2 ** self._t)
                if j == 0:
                    layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                else:
                    layer.b -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
