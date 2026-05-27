"""Unit tests for ``RBFLayer`` construction, forward pass, and gradients."""

import unittest
from engine.v1.value import Value
from engine.v1.layer import RBFLayer

class TestRBFLayer(unittest.TestCase):

    def test_layer_initialization_dimensions(self):
        """Verify that the edge matrix is created with the expected dimensions."""
        # 3 inputs, 5 outputs -> 5 rows and 3 edges per row.
        layer = RBFLayer(nin=3, nout=5)
        
        self.assertEqual(len(layer.out), 5, "Should have 5 output buckets")
        self.assertEqual(len(layer.out[0]), 3, "Each bucket should have 3 input edges")

    def test_parameter_collection(self):
        """Verify that all layer parameters are flattened into one list."""
        layer = RBFLayer(nin=2, nout=4)
        # 2 inputs * 4 outputs = 8 total edges. 
        # Each edge has 3 parameters (mean, width, amplitude). 
        # 8 * 3 = 24 total parameters.
        
        params = layer.parameters()
        
        self.assertEqual(len(params), 24, "Should have exactly 24 learnable parameters")
        
        # Verify that parameters are returned as Value instances.
        self.assertEqual(type(params[0]).__name__, "Value", "Parameters must be naked Value objects")

    def test_forward_pass_shape(self):
        """Verify output container type and output count."""
        layer = RBFLayer(nin=4, nout=2)
        
        # Create a representative input vector.
        x = [Value(1.0), Value(0.5), Value(-1.0), Value(2.0)]
        
        outputs = layer(x)
        
        self.assertEqual(type(outputs), list, "Layer must return a list")
        self.assertEqual(len(outputs), 2, "Layer should return exactly 2 output Values")
        self.assertEqual(type(outputs[0]).__name__, "Value", "Outputs must be Value objects")

    def test_backward_pass_cascade(self):
        """Verify that gradients propagate through the complete layer graph."""
        layer = RBFLayer(nin=2, nout=1)
        x = [Value(1.0), Value(-1.0)]
        
        # 1. Forward pass
        outputs = layer(x)
        
        # 2. Define a simple scalar loss
        loss = outputs[0] ** 2
        
        # 3. Backward pass
        loss.backward()
        
        # 4. Verify gradient flow
        params = layer.parameters()
        
        # If the graph is connected correctly, gradients reach all inputs and parameters.
        for p in params:
            self.assertNotEqual(p.gradient, 0.0, "Gradient did not reach layer parameter")
            
        self.assertNotEqual(x[0].gradient, 0.0, "Gradient did not reach input x[0]")
        self.assertNotEqual(x[1].gradient, 0.0, "Gradient did not reach input x[1]")

if __name__ == '__main__':
    unittest.main()