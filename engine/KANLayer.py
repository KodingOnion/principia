"""Single KAN layer implementing a batched RBF expansion.

The layer stores trainable center locations ``mu``, scale parameters
``gamma``, and per-center weights ``w``. Forward evaluation returns a
reduction over the center dimension followed by summation over input features.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.module import Module
from engine.tensor import Tensor
import numpy as np


class KANLayer(Module):
    def __init__(self, in_features, out_features, num_centers):
        grid_shape = (1, in_features, out_features, num_centers)

        base_centers = np.linspace(-2.0, 2.0, num_centers)
        self.mu = Tensor(np.ones(grid_shape) * base_centers)

        self.gamma = Tensor(np.ones(grid_shape) * 2.0)

        self.w = Tensor(np.random.randn(1, in_features, out_features, num_centers))

    def parameters(self):
        return [self.mu, self.gamma, self.w]

    def __call__(self, x):
        x_reshaped = x.reshape((x.data.shape[0], x.data.shape[1], 1, 1))

        activations = self.w * ((-self.gamma) * ((x_reshaped - self.mu) ** 2)).exp()

        activations = activations.sum(axis=3)
        activations = activations.sum(axis=1)

        return activations
