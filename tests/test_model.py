import unittest
from engine.value import Value
from engine.model import KAN

class TestKANModel(unittest.TestCase):

    def test_model_initialization(self):
        """Test if the layer sizes correctly translate into a stack of RBFLayers."""
        # A network with 3 inputs, two hidden layers (4 and 4 nodes), and 2 outputs
        model = KAN([3, 4, 4, 2])
        
        self.assertEqual(len(model.layers), 3, "Should have created exactly 3 layers")
        
        # Check dimensions of the first layer (3 in, 4 out)
        self.assertEqual(model.layers[0].nin, 3)
        self.assertEqual(model.layers[0].nout, 4)
        
        # Check dimensions of the final layer (4 in, 2 out)
        self.assertEqual(model.layers[-1].nin, 4)
        self.assertEqual(model.layers[-1].nout, 2)

    def test_forward_pass_cascade(self):
        """Test if the data successfully passes through the entire stack."""
        model = KAN([2, 5, 1])
        x = [Value(1.0), Value(-1.0)]
        
        out = model(x) # Trigger __call__
        
        self.assertEqual(type(out), list, "Model must return a list of outputs")
        self.assertEqual(len(out), 1, "Model should return exactly 1 output Value based on dimensions")
        self.assertEqual(type(out[0]).__name__, "Value", "Output must be a Value object")

    def test_master_parameter_harvesting(self):
        """Test if the model successfully collects parameters from every layer."""
        model = KAN([2, 3, 2])
        
        # Layer 1: 2 in, 3 out -> 6 edges * 3 params per edge = 18 params
        # Layer 2: 3 in, 2 out -> 6 edges * 3 params per edge = 18 params
        # Total parameters should be 36.
        
        params = model.parameters()
        
        self.assertEqual(len(params), 36, "Model failed to collect the correct total number of parameters")
        self.assertEqual(type(params[0]).__name__, "Value", "Parameters must be naked Value objects")

    def test_full_network_backpropagation(self):
        """The Ultimate Test: Does the chain rule survive the entire deep stack?"""
        model = KAN([2, 4, 1])
        x = [Value(0.5), Value(-0.5)]
        
        # 1. Forward pass
        out = model(x)
        loss = out[0] ** 2
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Check if gradients successfully flowed backwards through the 2D matrices
        params = model.parameters()
        for p in params:
            self.assertNotEqual(p.gradient, 0.0, "Gradient died before reaching a network parameter")
            
        self.assertNotEqual(x[0].gradient, 0.0, "Gradient did not reach network inputs")

if __name__ == '__main__':
    unittest.main()