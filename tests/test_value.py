"""Unit tests for the scalar autograd ``Value`` engine."""

import unittest
from engine.value import Value

class TestValueEngine(unittest.TestCase):

    def test_initialization(self):
        """Verify that Value initializes core attributes correctly."""
        v = Value(5.0)
        self.assertEqual(v.data, 5.0, "Data should store the float value.")
        # Gradient defaults to zero for leaf nodes.
        self.assertEqual(v.gradient, 0.0, "Gradient should default to 0.0")
        self.assertEqual(v.operation, None, "Default operation should be None")

    def test_addition_forward(self):
        """Verify forward computation for addition."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0, "2.0 + 3.0 should equal 5.0")
        self.assertEqual(c.operation, '+', "Operation string should be '+'")
        self.assertIn(a, c.children, "a should be a child of c")
        self.assertIn(b, c.children, "b should be a child of c")

    def test_multiplication_forward(self):
        """Verify forward computation for multiplication."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0, "2.0 * 3.0 should equal 6.0")
        self.assertEqual(c.operation, '*', "Operation string should be '*'")

    def test_addition_backward(self):
        """Verify backward propagation for addition."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        c = a + b
        
        # Simulate a downstream gradient signal.
        c.gradient = 1.0 
        c.backward()
        
        self.assertEqual(a.gradient, 1.0, "Derivative of (a+b) wrt a is 1.0 * c.grad")
        self.assertEqual(b.gradient, 1.0, "Derivative of (a+b) wrt b is 1.0 * c.grad")

    def test_multiplication_backward(self):
        """Verify backward propagation for multiplication."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        c = a * b
        
        # Simulate a downstream gradient signal.
        c.gradient = 1.0 
        c.backward()
        
        self.assertEqual(a.gradient, 3.0, "Derivative of (a*b) wrt a is b.data * c.grad")
        self.assertEqual(b.gradient, 2.0, "Derivative of (a*b) wrt b is a.data * c.grad")

    def test_gradient_accumulation(self):
        """Verify gradient accumulation when a node is reused."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        
        # ``a`` is used twice: c = (a * b) + a.
        # Therefore dc/da = b + 1. With b=3, dc/da=4.
        mult_node = a * b
        c = mult_node + a
        
        c.gradient = 1.0
        c.backward()
        
        self.assertEqual(a.gradient, 4.0, "Gradient must accumulate (+=), not overwrite (=)")

    def test_addition_with_constant(self):
        """Verify that Value can be added to a Python scalar."""
        a = Value(2.0)
        c = a + 5.0  # This should not crash
        self.assertEqual(c.data, 7.0, "Should be able to add a float to a Value object")

    def test_negation(self):
        """Verify unary negation behavior."""
        a = Value(3.0)
        c = -a
        self.assertEqual(c.data, -3.0, "Negation of 3.0 should be -3.0")

    def test_subtraction_forward_and_backward(self):
        """Verify subtraction forward result and gradients."""
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
        """Verify power forward result and derivative rule."""
        a = Value(2.0, gradient=0.0)
        c = a ** 3  # 2.0^3 = 8.0
        
        self.assertEqual(c.data, 8.0, "2.0 ** 3 should equal 8.0")
        
        c.gradient = 1.0
        c.backward()
        
        # d/da (a^3) = 3 * a^2 = 3 * 4 = 12
        self.assertEqual(a.gradient, 12.0, "Derivative of a^3 wrt a (where a=2) is 12.0")

    def test_division_forward_and_backward(self):
        """Verify division forward result and gradients."""
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
        """Verify exponential forward result and gradients."""
        import math
        a = Value(0.0, gradient=0.0)
        c = a.exp()
        
        # e^0 = 1.0
        self.assertEqual(c.data, 1.0, "e^0 should equal 1.0")
        
        c.gradient = 2.0
        c.backward()
        
        # d/da (e^a) = e^a. If a=0, e^a = 1.
        # Chain rule: local derivative * downstream gradient = 1 * 2 = 2.
        self.assertEqual(a.gradient, 2.0, "Derivative of e^a wrt a is e^a * downstream_gradient")

    def test_complex_expression(self):
        """Verify gradients for a multi-step composed expression."""
        a = Value(2.0, gradient=0.0)
        b = Value(3.0, gradient=0.0)
        
        # Equation: c = (a ** 2) / b  -> (4 / 3 = 1.333...)
        c = (a ** 2) / b
        
        # Expected gradients:
        # dc/da = (2*a) / b = 4/3 = 1.333...
        # dc/db = -(a**2) / (b**2) = -4 / 9 = -0.444...
        
        c.gradient = 1.0
        c.backward()
        
        self.assertAlmostEqual(a.gradient, 4/3, places=4, msg="Complex chain rule failed for a")
        self.assertAlmostEqual(b.gradient, -4/9, places=4, msg="Complex chain rule failed for b")

if __name__ == '__main__':
    unittest.main()