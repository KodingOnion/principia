import unittest
from engine.value import Value
from engine.layer import RBFLayer

class TestRBFLayer(unittest.TestCase):

    def test_layer_initialization_dimensions(self):
        """Test if the 2D matrix of edges is built with the correct shape."""
        # 3 inputs, 5 outputs = 5 buckets, 3 edges per bucket
        layer = RBFLayer(nin=3, nout=5)
        
        self.assertEqual(len(layer.out), 5, "Should have 5 output buckets")
        self.assertEqual(len(layer.out[0]), 3, "Each bucket should have 3 input edges")

    def test_parameter_collection(self):
        """Test if the layer correctly flattens all edge parameters into a 1D list."""
        layer = RBFLayer(nin=2, nout=4)
        # 2 inputs * 4 outputs = 8 total edges. 
        # Each edge has 3 parameters (mean, width, amplitude). 
        # 8 * 3 = 24 total parameters.
        
        params = layer.parameters()
        
        self.assertEqual(len(params), 24, "Should have exactly 24 learnable parameters")
        
        # Verify the first item is actually a Value object
        self.assertEqual(type(params[0]).__name__, "Value", "Parameters must be naked Value objects")

    def test_forward_pass_shape(self):
        """Test if a forward pass produces the correct number of outputs."""
        layer = RBFLayer(nin=4, nout=2)
        
        # Create a dummy input list of 4 Value objects
        x = [Value(1.0), Value(0.5), Value(-1.0), Value(2.0)]
        
        outputs = layer(x) # Trigger __call__
        
        self.assertEqual(type(outputs), list, "Layer must return a list")
        self.assertEqual(len(outputs), 2, "Layer should return exactly 2 output Values")
        self.assertEqual(type(outputs[0]).__name__, "Value", "Outputs must be Value objects")

    def test_backward_pass_cascade(self):
        """Test if the master DAG connects the entire layer for backprop."""
        layer = RBFLayer(nin=2, nout=1)
        x = [Value(1.0), Value(-1.0)]
        
        # 1. Forward Pass
        outputs = layer(x)
        
        # 2. Simulate a "Loss" function by squaring the output
        loss = outputs[0] ** 2
        
        # 3. Trigger the Topological Sort
        loss.backward()
        
        # 4. Verify gradients flowed backwards
        params = layer.parameters()
        
        # If the DAG is unbroken, every single parameter and input should have a non-zero gradient.
        for p in params:
            self.assertNotEqual(p.gradient, 0.0, "Gradient did not reach layer parameter")
            
        self.assertNotEqual(x[0].gradient, 0.0, "Gradient did not reach input x[0]")
        self.assertNotEqual(x[1].gradient, 0.0, "Gradient did not reach input x[1]")

if __name__ == '__main__':
    unittest.main()