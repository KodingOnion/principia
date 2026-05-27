import unittest
from engine.v1.value import Value
from engine.v1.optim import AdamOptimizer

class TestAdamOptimizer(unittest.TestCase):

    def test_optimizer_initialization(self):
        """Test if the memory ledgers and time step initialize correctly."""
        p1 = Value(1.0)
        p2 = Value(-1.0)
        
        optimizer = AdamOptimizer([p1, p2], learning_rate=0.01)
        
        self.assertEqual(optimizer.t, 0, "Time step should initialize to 0")
        self.assertEqual(len(optimizer.m), 2, "Momentum ledger should have an entry for each parameter")
        self.assertEqual(len(optimizer.v), 2, "Variance ledger should have an entry for each parameter")
        
        # Check that the starting memory is exactly 0.0
        self.assertEqual(optimizer.m[p1], 0.0)
        self.assertEqual(optimizer.v[p2], 0.0)

    def test_zero_grad(self):
        """Test if the optimizer successfully wipes the gradients of all tracked parameters."""
        p1 = Value(1.0, gradient=5.5)
        p2 = Value(2.0, gradient=-3.14)
        
        optimizer = AdamOptimizer([p1, p2])
        optimizer.zero_grad()
        
        self.assertEqual(p1.gradient, 0.0, "p1 gradient was not zeroed")
        self.assertEqual(p2.gradient, 0.0, "p2 gradient was not zeroed")

    def test_adam_step_math(self):
        """The Ultimate Test: Hand-calculate the first Adam step to ensure exact precision."""
        # Setup: A parameter at 1.0 with a gradient of 0.1
        p = Value(1.0, gradient=0.1)
        optimizer = AdamOptimizer([p], learning_rate=0.1)
        
        # Trigger the step
        optimizer.step()

        # m_new = 0.9*0 + 0.1*0.1 = 0.01
        # v_new = 0.999*0 + 0.001*(0.1^2) = 0.00001
        
        # Bias corrections:
        # m_hat = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
        # v_hat = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
        
        # Update step:
        # step = learning_rate * (m_hat / (sqrt(v_hat) + 1e-8))
        # step = 0.1 * (0.1 / (0.1 + 1e-8)) ≈ 0.1 * 1.0 = 0.1
        
        # New data: 1.0 - 0.1 = 0.9
        
        self.assertEqual(optimizer.t, 1, "Time step did not increment")
        self.assertAlmostEqual(optimizer.m[p], 0.01, places=5, msg="Momentum math is incorrect")
        self.assertAlmostEqual(optimizer.v[p], 0.00001, places=7, msg="Variance math is incorrect")
        
        # The new parameter value should be extremely close to 0.9
        self.assertAlmostEqual(p.data, 0.9, places=5, msg="Final weight update math is incorrect")

if __name__ == '__main__':
    unittest.main()