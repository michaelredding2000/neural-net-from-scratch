"""Loss functions."""

import numpy as np


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Softmax cross-entropy.

    Parameters
    ----------
    probs : (batch, n_classes) softmax probabilities
    y     : (batch,) integer class labels

    Returns
    -------
    scalar mean loss
    """
    batch = probs.shape[0]
    clipped = np.clip(probs[np.arange(batch), y], 1e-15, 1.0)
    return -np.mean(np.log(clipped))


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of softmax cross-entropy w.r.t. the pre-softmax logits.
    Shape: (batch, n_classes)
    """
    batch = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(batch), y] -= 1
    return grad / batch


def mse_loss(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))


def mse_grad(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return 2 * (preds - targets) / preds.shape[0]
