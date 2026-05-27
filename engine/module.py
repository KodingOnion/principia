"""Base neural module providing utilities for parameter management."""

import numpy as np


class Module:
    """Abstract base class for models and layers.

    Subclasses should override :meth:`parameters` to return an iterable of
    trainable ``Tensor`` instances.
    """

    def zero_grad(self):
        """Reset gradients for every tracked parameter to zeros."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        """Return an iterable of trainable parameters. Override in subclasses."""
        return []