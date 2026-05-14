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

    def test_negation(self):
        """Test if negation correctly flips the sign."""
        a = Value(3.0)
        c = -a
        self.assertEqual(c.data, -3.0, "Negation of 3.0 should be -3.0")

    def test_subtraction_forward_and_backward(self):
        """Test subtraction forward math and chain rule."""
        a = Value(5.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        c = a - b
        
        self.assertEqual(c.data, 2.0, "5.0 - 3.0 should equal 2.0")
        
        c.gradient = 1.0
        c.backward()
        
        # d/da (a - b) = 1
        # d/db (a - b) = -1
        self.assertEqual(a.gradient, 1.0, "Derivative of (a-b) wrt a is 1.0")
        self.assertEqual(b.gradient, -1.0, "Derivative of (a-b) wrt b is -1.0")

    def test_power_forward_and_backward(self):
        """Test power rule (x^n) forward math and chain rule."""
        a = Value(2.0, gradient=0.0)
        c = a ** 3  # 2.0^3 = 8.0
        
        self.assertEqual(c.data, 8.0, "2.0 ** 3 should equal 8.0")
        
        c.gradient = 1.0
        c.backward()
        
        # d/da (a^3) = 3 * a^2 = 3 * 4 = 12
        self.assertEqual(a.gradient, 12.0, "Derivative of a^3 wrt a (where a=2) is 12.0")

    def test_division_forward_and_backward(self):
        """Test division forward math and quotient/power chain rule."""
        a = Value(6.0, gradient=0.0)
        b = Value(2.0, gradient=0.0)
        c = a / b
        
        self.assertEqual(c.data, 3.0, "6.0 / 2.0 should equal 3.0")
        
        c.gradient = 1.0
        c.backward()
        
        # c = a * b^-1
        # d/da = 1/b = 1/2 = 0.5
        # d/db = -a / (b^2) = -6 / 4 = -1.5
        self.assertEqual(a.gradient, 0.5, "Derivative of (a/b) wrt a is 1/b")
        self.assertEqual(b.gradient, -1.5, "Derivative of (a/b) wrt b is -a/b^2")

    def test_exponential_forward_and_backward(self):
        """Test e^x forward math and chain rule."""
        import math
        a = Value(0.0, gradient=0.0)
        c = a.exp()
        
        # e^0 = 1.0
        self.assertEqual(c.data, 1.0, "e^0 should equal 1.0")
        
        c.gradient = 2.0 # Passing a gradient of 2.0 from "downstream"
        c.backward()
        
        # d/da (e^a) = e^a. If a=0, e^a = 1. 
        # Chain rule: local_derivative * downstream_gradient = 1.0 * 2.0 = 2.0
        self.assertEqual(a.gradient, 2.0, "Derivative of e^a wrt a is e^a * downstream_gradient")

    def test_complex_expression(self):
        """Test a multi-step expression to ensure the graph chains correctly."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        
        # Equation: c = (a ** 2) / b  -> (4 / 3 = 1.333...)
        c = (a ** 2) / b
        
        # Let's check the gradients. 
        # dc/da = (2*a) / b = 4/3 = 1.333...
        # dc/db = -(a**2) / (b**2) = -4 / 9 = -0.444...
        
        c.gradient = 1.0
        c.backward() # This will only trigger the division node!
        
        # Wait! Because you don't have the topological sort built yet, 
        # calling c.backward() won't automatically cascade down to the power node.
        # So we have to manually trigger the child node for this test to work right now.
        for child in c.children:
            if child.operation == '**': # The (a**2) node
                child.backward()
        
        self.assertAlmostEqual(a.gradient, 4/3, places=4, msg="Complex chain rule failed for a")
        self.assertAlmostEqual(b.gradient, -4/9, places=4, msg="Complex chain rule failed for b")

if __name__ == '__main__':
    unittest.main()