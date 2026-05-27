import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.module import Module
from engine.tensor import Tensor
import numpy as np


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.w = Tensor(np.random.randn(in_features, out_features))
        self.b = Tensor(np.zeros((1, out_features)))

    def __call__(self, x):
        return (x @ self.w) + self.b
    
    def parameters(self):
        return [self.w,self.b]