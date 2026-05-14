import unittest
import math
from engine.value import Value
from engine.rbf_edge import RBFEdge as rbf_edge

class TestRBFEdge(unittest.TestCase):

    def test_initialization_type_promotion(self):
        """Test if the edge initializes raw floats into Value objects."""
        # We pass raw floats, but the class should promote them to Value objects
        edge = rbf_edge(mean=0.5, width=1.0, amplitude=2.0)
        
        self.assertEqual(type(edge.mean).__name__, "Value", "Mean must be wrapped in a Value object")
        self.assertEqual(type(edge.width).__name__, "Value", "Width must be wrapped in a Value object")
        self.assertEqual(type(edge.amplitude).__name__, "Value", "Amplitude must be wrapped in a Value object")

    def test_forward_pass_math(self):
        """Test the exact math of the Gaussian equation."""
        edge = rbf_edge(mean=0.0, width=1.0, amplitude=2.0)
        x = Value(1.0)
        
        y = edge(x) # This triggers __call__

        # Math: 2.0 * exp(-1.0 * (1.0 - 0.0)^2) = 2.0 * exp(-1.0)
        expected_val = 2.0 * math.exp(-1.0)
        self.assertAlmostEqual(y.data, expected_val, places=4, msg="Forward Gaussian math is incorrect")

    def test_backward_pass_calculus(self):
        """Test the chain rule flowing through the entire Gaussian equation."""
        edge = rbf_edge(mean=0.5, width=1.0, amplitude=2.0)
        x = Value(1.5)

        y = edge(x)
        y.backward() # Trigger the master topological sort

        # Let's calculate the expected calculus by hand:
        # y = w * exp(-g * (x - m)^2)
        # x = 1.5, m = 0.5 -> (x-m) = 1.0
        # g = 1.0 -> -g*(x-m)^2 = -1.0
        
        expected_dw = math.exp(-1.0)
        expected_dg = 2.0 * math.exp(-1.0) * -1.0
        expected_dm = 2.0 * math.exp(-1.0) * -1.0 * 2.0 * (1.0) * -1.0
        expected_dx = 2.0 * math.exp(-1.0) * -1.0 * 2.0 * (1.0)

        self.assertAlmostEqual(edge.amplitude.gradient, expected_dw, places=4, msg="Amplitude gradient incorrect")
        self.assertAlmostEqual(edge.width.gradient, expected_dg, places=4, msg="Width gradient incorrect")
        self.assertAlmostEqual(edge.mean.gradient, expected_dm, places=4, msg="Mean gradient incorrect")
        self.assertAlmostEqual(x.gradient, expected_dx, places=4, msg="Input X gradient incorrect")

    def test_type_checking(self):
        """Test if __call__ correctly rejects non-Value inputs."""
        edge = rbf_edge()
        with self.assertRaises(TypeError):
            edge(5.0) # Should crash because 5.0 is a float, not a Value object

if __name__ == '__main__':
    unittest.main()