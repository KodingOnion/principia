"""Unit tests for the Tensor autograd engine."""

import unittest
import numpy as np
from principia import Tensor, mse_loss


class TestTensorAutograd(unittest.TestCase):

    # --- ADDITION TESTS ---

    def test_simple_addition_forward_and_backward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a + b

        np.testing.assert_array_equal(c.data, np.array([[6.0, 8.0], [10.0, 12.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad, np.ones_like(b.data))

    def test_addition_with_python_scalar(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a + 5.0

        np.testing.assert_array_equal(c.data, np.array([[6.0, 7.0], [8.0, 9.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        wrapped_scalar = [child for child in c._prev if child is not a][0]
        self.assertEqual(wrapped_scalar.grad.shape, ())
        self.assertEqual(wrapped_scalar.grad, 4.0)

    def test_broadcasting_vector_to_matrix(self):
        batch = Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        bias = Tensor([10.0, 20.0])
        out = batch + bias

        np.testing.assert_array_equal(
            out.data, np.array([[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])
        )

        # Simulating a specific incoming gradient to test broadcasting math
        out.grad = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        # We temporarily bypass the root 1s initialization for this specific manual test
        # Force the local un-broadcasting test
        out._backward()

        np.testing.assert_array_equal(bias.grad, np.array([3.0, 6.0]))

    def test_broadcasting_stretched_dimension(self):
        a = Tensor([[1.0], [2.0], [3.0]])
        b = Tensor([[10.0, 20.0]])
        c = a + b

        np.testing.assert_array_equal(
            c.data, np.array([[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])
        )
        c.backward()

        np.testing.assert_array_equal(a.grad, np.array([[2.0], [2.0], [2.0]]))
        np.testing.assert_array_equal(b.grad, np.array([[3.0, 3.0]]))

    # --- POWER TESTS ---

    def test_power_forward_and_backward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a**2

        np.testing.assert_array_equal(c.data, np.array([[1.0, 4.0], [9.0, 16.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.array([[2.0, 4.0], [6.0, 8.0]]))

    def test_power_negative_exponent(self):
        a = Tensor([2.0, 4.0])
        c = a**-1

        np.testing.assert_array_equal(c.data, np.array([0.5, 0.25]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.array([-0.25, -0.0625]))

    def test_power_fractional_exponent(self):
        a = Tensor([4.0, 9.0])
        c = a**0.5

        np.testing.assert_array_equal(c.data, np.array([2.0, 3.0]))
        c.backward()

        np.testing.assert_array_almost_equal(
            a.grad, np.array([0.25, 0.16666667]), decimal=5
        )

    # --- MULTIPLICATION TESTS ---

    def test_multiplication_forward_and_backward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[2.0, 3.0], [4.0, 5.0]])
        c = a * b

        np.testing.assert_array_equal(c.data, np.array([[2.0, 6.0], [12.0, 20.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, b.data)
        np.testing.assert_array_equal(b.grad, a.data)

    def test_multiplication_with_python_scalar(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a * 3.0

        np.testing.assert_array_equal(c.data, np.array([[3.0, 6.0], [9.0, 12.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.full(a.data.shape, 3.0))

    def test_multiplication_broadcasting(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 100.0])
        c = a * b

        np.testing.assert_array_equal(c.data, np.array([[10.0, 200.0], [30.0, 400.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.array([[10.0, 100.0], [10.0, 100.0]]))
        np.testing.assert_array_equal(b.grad, np.array([4.0, 6.0]))

    # --- DEPENDENT OPERATIONS (SUB, DIV, NEG) ---

    def test_negation_forward_and_backward(self):
        a = Tensor([[1.0, -2.0], [3.0, 0.0]])
        c = -a

        np.testing.assert_array_equal(c.data, np.array([[-1.0, 2.0], [-3.0, 0.0]]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.full(a.data.shape, -1.0))

    def test_subtraction_forward_and_backward(self):
        a = Tensor([5.0, 10.0])
        b = Tensor([2.0, 3.0])
        c = a - b

        np.testing.assert_array_equal(c.data, np.array([3.0, 7.0]))
        c.backward()

        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad, np.full(b.data.shape, -1.0))

    def test_division_forward_and_backward(self):
        a = Tensor([6.0, 12.0])
        b = Tensor([2.0, 3.0])
        c = a / b

        np.testing.assert_array_equal(c.data, np.array([3.0, 4.0]))
        c.backward()

        np.testing.assert_array_almost_equal(
            a.grad, np.array([0.5, 0.33333333]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            b.grad, np.array([-1.5, -1.33333333]), decimal=5
        )

    # --- MATRIX MULTIPLICATION TESTS ---

    def test_matmul_square_matrices(self):
        """Verify the matrix calculus (A @ B) -> dA = dC @ B.T, dB = A.T @ dC."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])

        c = a @ b

        # Forward pass: row dot column
        np.testing.assert_array_equal(c.data, np.array([[19.0, 22.0], [43.0, 50.0]]))

        c.backward()

        # dC is an array of 1s: [[1, 1], [1, 1]]
        # da = dC @ b.T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
        np.testing.assert_array_equal(a.grad, np.array([[11.0, 15.0], [11.0, 15.0]]))

        # db = a.T @ dC = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        np.testing.assert_array_equal(b.grad, np.array([[4.0, 4.0], [6.0, 6.0]]))

    def test_matmul_neural_network_layer_shape(self):
        """Verify batched input multiplied by a weight matrix (Batch, In) @ (In, Out)."""
        # Batch of 3 inputs, 2 features each (3, 2)
        a = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Weight matrix mapping 2 inputs to 3 outputs (2, 3)
        b = Tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])

        c = a @ b

        # Expected shape is (3, 3)
        expected_out = np.array([[1.0, 2.0, -3.0], [3.0, 4.0, -7.0], [5.0, 6.0, -11.0]])
        np.testing.assert_array_equal(c.data, expected_out)

        c.backward()

        # da = dC (3x3 of 1s) @ b.T (3x2) -> result is (3x2)
        expected_da = np.zeros((3, 2))  # Since sum of columns in b.T is 0
        np.testing.assert_array_equal(a.grad, expected_da)

        # db = a.T (2x3) @ dC (3x3 of 1s) -> result is (2x3)
        expected_db = np.array([[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]])
        np.testing.assert_array_equal(b.grad, expected_db)

    def test_sum_forward_and_backward(self):
        """Verify the sum operation and its gradient distribution."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a.sum()

        # Forward pass: 1 + 2 + 3 + 4 = 10
        self.assertEqual(c.data, 10.0)

        c.backward()

        # Backward pass: gradient of sum is 1.0 distributed to every element
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))

    # --- MISSING TENSOR OPS (RESHAPE & EXP) ---
    
    def test_reshape_forward_and_backward(self):
        a = Tensor([[1.0, 2.0, 3.0, 4.0]])
        b = a.reshape((2, 2))
        
        np.testing.assert_array_equal(b.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
        b.backward()
        
        # Gradient should reshape perfectly back to the original 1x4
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))

    def test_exp_forward_and_backward(self):
        a = Tensor([1.0, 2.0])
        b = a.exp()
        
        # Forward pass: e^1, e^2
        np.testing.assert_array_almost_equal(b.data, np.array([2.7182818, 7.3890561]), decimal=5)
        b.backward()
        
        # Backward pass: The local derivative of e^x is e^x. 
        np.testing.assert_array_almost_equal(a.grad, b.data, decimal=5)

    def test_mse_loss_forward_and_backward(self):
        # Assuming mse_loss is imported or defined in the test file
        predictions = Tensor([2.0, 4.0, 6.0])
        targets = Tensor([1.0, 4.0, 8.0])
        
        # Diff: [1.0, 0.0, -2.0] -> Sq: [1.0, 0.0, 4.0] -> Sum: 5.0 -> Mean: 1.666...
        loss = mse_loss(predictions, targets)
        np.testing.assert_almost_equal(loss.data, 1.6666667, decimal=5)
        
        loss.backward()
        # d/dx of MSE = 2 * (preds - targets) / n
        expected_grad = np.array([0.6666666, 0.0, -1.3333333])
        np.testing.assert_array_almost_equal(predictions.grad, expected_grad, decimal=5)
    
if __name__ == "__main__":
    unittest.main()
