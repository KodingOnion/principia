"""High-level KAN model composed of stacked KANLayer modules.

This module provides a small wrapper to construct a network from a list of
layer sizes and expose a simple functional API for forward evaluation and
parameter harvesting.
"""

from engine.module import Module
from engine.KANLayer import KANLayer
import itertools


class KAN(Module):
    """Kolmogorov-Arnold Network: a stack of radial basis layers.

    Parameters
    - layer_sizes: iterable of ints describing layer widths
    - num_centers: number of RBF centers per input (optional)
    """

    def __init__(self, layer_sizes, num_centers=None):
        self.layer_sizes = layer_sizes
        self.num_centers = num_centers if num_centers is not None else 5

        self.layers = []

        for in_feat, out_feat in itertools.pairwise(layer_sizes):
            self.layers.append(KANLayer(in_feat, out_feat, num_centers))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            for param in layer.parameters():
                params.append(param)
        return params