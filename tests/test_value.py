import unittest
from engine.value import Value

class TestValueEngine(unittest.TestCase):

    def test_initialization(self):
        """Test if the Value object initializes attributes correctly."""
        v = Value(5.0)
        self.assertEqual(v.data, 5.0, "Data should store the float value.")
        # If no gradient is provided, it's usually best practice for it to default to 0.0
        self.assertEqual(v.gradient, 0.0, "Gradient should default to 0.0")
        self.assertEqual(v.operation, None, "Default operation should be None")

    def test_addition_forward(self):
        """Test if the forward pass of addition computes the correct data."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0, "2.0 + 3.0 should equal 5.0")
        self.assertEqual(c.operation, '+', "Operation string should be '+'")
        self.assertIn(a, c.children, "a should be a child of c")
        self.assertIn(b, c.children, "b should be a child of c")

    def test_multiplication_forward(self):
        """Test if the forward pass of multiplication computes the correct data."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0, "2.0 * 3.0 should equal 6.0")
        self.assertEqual(c.operation, '*', "Operation string should be '*'")

    def test_addition_backward(self):
        """Test if the chain rule for addition passes gradients back correctly."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        c = a + b
        
        # Simulate a gradient arriving from further down the network
        c.gradient = 1.0 
        c.backward()
        
        self.assertEqual(a.gradient, 1.0, "Derivative of (a+b) wrt a is 1.0 * c.grad")
        self.assertEqual(b.gradient, 1.0, "Derivative of (a+b) wrt b is 1.0 * c.grad")

    def test_multiplication_backward(self):
        """Test if the chain rule for multiplication passes gradients back correctly."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        c = a * b
        
        # Simulate a gradient arriving from further down the network
        c.gradient = 1.0 
        c.backward()
        
        self.assertEqual(a.gradient, 3.0, "Derivative of (a*b) wrt a is b.data * c.grad")
        self.assertEqual(b.gradient, 2.0, "Derivative of (a*b) wrt b is a.data * c.grad")

    def test_gradient_accumulation(self):
        """CRITICAL: Test if a variable used multiple times adds its gradients together."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        
        # 'a' is used twice here. c = (a * b) + a
        # Mathematically, dc/da = b + 1. If b=3, dc/da should be 4.
        mult_node = a * b
        c = mult_node + a
        
        c.gradient = 1.0
        c.backward()          # Run addition backward
        mult_node.backward()  # Run multiplication backward
        
        self.assertEqual(a.gradient, 4.0, "Gradient must accumulate (+=), not overwrite (=)")

    def test_addition_with_constant(self):
        """Test if the Value class can handle being added to a normal Python float."""
        a = Value(2.0)
        c = a + 5.0  # This should not crash
        self.assertEqual(c.data, 7.0, "Should be able to add a float to a Value object")

if __name__ == '__main__':
    unittest.main()