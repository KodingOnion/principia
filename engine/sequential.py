"""High-level KAN model composed of stacked KANLayer modules.

This module provides a small wrapper to construct a network from a list of
layer sizes and expose a simple functional API for forward evaluation and
parameter harvesting.
"""

from engine.module import Module
import pickle

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
    
    def save_weights(self, filepath):
        """Extracts and saves only the raw NumPy arrays to avoid autograd bloat."""
        # Grab the pure data, leaving the Tensor wrappers and graphs behind
        weights = [param.data for param in self.parameters()]
        
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Weights successfully saved to {filepath}")

    def load_weights(self, filepath):
        """Overwrites the current network parameters with saved weights."""
        with open(filepath, 'rb') as f:
            saved_weights = pickle.load(f)
            
        current_params = self.parameters()
        
        # Ensure the model architecture exactly matches the saved weights
        if len(current_params) != len(saved_weights):
            raise ValueError("Architecture mismatch: Parameter counts do not align.")
            
        # Overwrite the data matrix of each tensor in place
        for param, saved_weight in zip(current_params, saved_weights):
            if param.data.shape != saved_weight.shape:
                 raise ValueError(f"Shape mismatch: expected {param.data.shape}, got {saved_weight.shape}")
            param.data = saved_weight
            
        print(f"Weights successfully loaded from {filepath}")