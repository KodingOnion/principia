import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.module import Module
from engine.tensor import Tensor
from engine.KANLayer import KANLayer
import numpy as np
import itertools

class KAN(Module):
    def __init__(self,layer_sizes,num_centers=None):
        self.layer_sizes = layer_sizes
        self.num_centers = num_centers if not None else 5

        self.layers = []

        for in_feat, out_feat in itertools.pairwise(layer_sizes):
            self.layers.append(KANLayer(in_feat, out_feat, num_centers))

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)

        return x
    
    def parameters(self):
        params = []

        for layer in self.layers:
            for param in layer.parameters():
                params.append(param)

        return params