import unittest
import numpy as np
from principia import Tensor, Linear, RBFLayer, Sequential, ChebLayer

class TestNeuralNetworkArchitecture(unittest.TestCase):

    # --- LINEAR LAYER TESTS ---

    def test_linear_layer_shapes_and_params(self):
        model = Linear(in_features=3, out_features=5)
        x = Tensor(np.random.randn(4, 3)) # Batch of 4, 3 features
        
        out = model(x)
        
        # Check output geometry (Batch, Out_Features)
        self.assertEqual(out.data.shape, (4, 5))
        
        # Check parameter fetching (should have 2 parameters: w and b)
        params = model.parameters()
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].data.shape, (3, 5)) # Weights
        self.assertEqual(params[1].data.shape, (1, 5)) # Bias

    def test_zero_grad_functionality(self):
        model = Linear(2, 2)
        x = Tensor([[1.0, 2.0]])
        
        # Fake a training step
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Verify gradients are populated
        self.assertTrue(np.any(model.w.grad != 0.0))
        
        # Wipe them
        model.zero_grad()
        
        # Verify gradients are completely zeroed
        np.testing.assert_array_equal(model.w.grad, np.zeros_like(model.w.grad))
        np.testing.assert_array_equal(model.b.grad, np.zeros_like(model.b.grad))


    # --- RBF LAYER TESTS ---

    def test_rbf_layer_forward_geometry(self):
        in_feat, out_feat, centers = 2, 4, 10
        model = RBFLayer(in_features=in_feat, out_features=out_feat, num_centers=centers)
        
        # Batch of 3
        x = Tensor(np.random.randn(3, in_feat)) 
        out = model(x)
        
        # Despite massive 4D internal grids, output should be standard 2D
        self.assertEqual(out.data.shape, (3, out_feat))
        
        # Check params (mu, gamma, w)
        params = model.parameters()
        self.assertEqual(len(params), 3)
        
        expected_grid_shape = (1, in_feat, out_feat, centers)
        for param in params:
            self.assertEqual(param.data.shape, expected_grid_shape)


    # --- SEQUENTIAL WRAPPER TESTS ---

    def test_sequential_network_sequence_and_aggregation(self):
        centers = 10
        # Initialize layers individually and pass them in a list
        layer1 = RBFLayer(in_features=2, out_features=5, num_centers=centers)
        layer2 = RBFLayer(in_features=5, out_features=3, num_centers=centers)
        
        model = Sequential([layer1, layer2])
        
        # Should contain exactly 2 layers
        self.assertEqual(len(model.layers), 2)
        
        # Test full sequential forward pass
        x = Tensor(np.random.randn(8, 2)) # Batch of 8
        out = model(x)
        
        self.assertEqual(out.data.shape, (8, 3))
        
        # Test parameter aggregation
        # 2 layers * 3 params per layer (mu, gamma, w) = 6 total parameter tensors
        params = model.parameters()
        self.assertEqual(len(params), 6)

    # --- CHEBYSHEV LAYER TESTS ---

    def test_cheb_layer_initialization_and_params(self):
        """Verify the layer sets up the correct list of coefficient tensors."""
        in_feat, out_feat, degree = 3, 5, 4
        model = ChebLayer(in_features=in_feat, out_features=out_feat, degree=degree)
        
        params = model.parameters()
        
        # The optimizer needs exactly 'degree' number of parameter tensors
        self.assertEqual(len(params), degree)
        
        # Each tensor in the list must match the broadcasting geometry
        expected_shape = (1, in_feat, out_feat)
        for param in params:
            self.assertEqual(param.data.shape, expected_shape)

    def test_cheb_layer_forward_geometry(self):
        """Verify the polynomial generation collapses correctly into a 2D batch."""
        in_feat, out_feat, degree = 2, 4, 5
        model = ChebLayer(in_features=in_feat, out_features=out_feat, degree=degree)
        
        # Batch of 6 points
        x = Tensor(np.random.randn(6, in_feat)) 
        out = model(x)
        
        # Despite generating 5 different 3D tensors internally, the final
        # result must perfectly collapse back to (Batch, Out_Features)
        self.assertEqual(out.data.shape, (6, out_feat))

    def test_cheb_layer_backward_pass(self):
        """Verify the computation graph remains fully connected through the list loop."""
        model = ChebLayer(in_features=2, out_features=3, degree=4)
        
        # Feed a simple batch and force a gradient update
        x = Tensor([[0.5, -0.5], [0.1, 0.9]])
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check every single degree's coefficient tensor
        params = model.parameters()
        for param in params:
            # If any tensor's gradient is perfectly zero, the chain rule broke!
            self.assertTrue(np.any(param.grad != 0.0), 
                            "Gradient failed to flow into the coefficient tensor.")
        
    
if __name__ == '__main__':
    unittest.main()