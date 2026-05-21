"""Unit tests for the Tensor autograd engine."""

import unittest
import numpy as np
from engine.tensor import Tensor 

class TestTensorAddition(unittest.TestCase):

    def test_simple_addition_forward_and_backward(self):
        """Verify addition with identical matrix shapes."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        
        c = a + b
        
        # Check forward pass
        np.testing.assert_array_equal(c.data, np.array([[6.0, 8.0], [10.0, 12.0]]))
        
        # Simulate an incoming gradient matrix of 1s
        c.grad = np.ones_like(c.data)
        c._backward()
        
        # Gradients should flow backward equally
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad, np.ones_like(b.data))

    def test_addition_with_python_scalar(self):
        """Verify that pure Python numbers are wrapped and broadcasted."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        
        c = a + 5.0
        
        # Check forward pass
        np.testing.assert_array_equal(c.data, np.array([[6.0, 7.0], [8.0, 9.0]]))
        
        # Simulate incoming gradient
        c.grad = np.ones_like(c.data)
        c._backward()
        
        # The gradient for 'a' should be all 1s.
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        
        # The 5.0 was wrapped in a Tensor and broadcasted from shape () to (2, 2).
        # Its gradient should be the sum of the entire matrix (4.0).
        # We find that wrapped tensor in c's children.
        wrapped_scalar = [child for child in c._prev if child is not a][0]
        self.assertEqual(wrapped_scalar.grad.shape, ())
        self.assertEqual(wrapped_scalar.grad, 4.0)

    def test_broadcasting_vector_to_matrix(self):
        """Verify gradient un-broadcasting when a 1D vector is added to a 2D matrix."""
        # Batch of 3 inputs, each with 2 features (3, 2)
        batch = Tensor([[1.0, 1.0], 
                        [2.0, 2.0], 
                        [3.0, 3.0]])
        
        # Bias vector for the 2 features (2,)
        bias = Tensor([10.0, 20.0])
        
        out = batch + bias
        
        # Check forward broadcasted math
        expected_out = np.array([[11.0, 21.0], 
                                 [12.0, 22.0], 
                                 [13.0, 23.0]])
        np.testing.assert_array_equal(out.data, expected_out)
        
        # Simulate a specific incoming gradient pattern
        out.grad = np.array([[1.0, 2.0], 
                             [1.0, 2.0], 
                             [1.0, 2.0]])
        out._backward()
        
        # Batch gradient should match output gradient exactly
        np.testing.assert_array_equal(batch.grad, out.grad)
        
        # Bias gradient must be summed across axis 0. 
        # (1+1+1=3, 2+2+2=6)
        np.testing.assert_array_equal(bias.grad, np.array([3.0, 6.0]))

    def test_broadcasting_stretched_dimension(self):
        """Verify un-broadcasting when an internal dimension is stretched."""
        # Shape (3, 1)
        a = Tensor([[1.0], [2.0], [3.0]])
        # Shape (1, 2)
        b = Tensor([[10.0, 20.0]])
        
        c = a + b
        
        # Check forward math (results in a 3x2 matrix)
        expected_out = np.array([[11.0, 21.0],
                                 [12.0, 22.0],
                                 [13.0, 23.0]])
        np.testing.assert_array_equal(c.data, expected_out)
        
        c.grad = np.ones_like(c.data)
        c._backward()
        
        # 'a' was stretched along axis 1. Gradient should sum to 2s.
        np.testing.assert_array_equal(a.grad, np.array([[2.0], [2.0], [2.0]]))
        
        # 'b' was stretched along axis 0. Gradient should sum to 3s.
        np.testing.assert_array_equal(b.grad, np.array([[3.0, 3.0]]))

if __name__ == '__main__':
    unittest.main()