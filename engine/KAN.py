"""High-level KAN model composed of stacked KANLayer modules.

This module provides a small wrapper to construct a network from a list of
layer sizes and expose a simple functional API for forward evaluation and
parameter harvesting.
"""

from engine.module import Module

class Sequential(Module):
    """Kolmogorov-Arnold Network: a stack of radial basis layers.

    Parameters
    - layers: a list of instantiated layer modules.
    """

    def __init__(self, layers):
        self.layers = layers

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