"""Simple Adam optimizer wrapper for lists of Tensor-like parameters."""

import numpy as np


class AdamOptimiser:
    """Maintain Adam moment estimates and apply parameter updates.

    The optimiser stores first- and second-moment estimates per-parameter and
    performs standard bias-corrected Adam updates in :meth:`step`.
    """

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

        self.m = {}
        self.v = {}
        self.t = 0

        for parameter in parameters:
            self.m[parameter] = np.zeros_like(parameter.data)
            self.v[parameter] = np.zeros_like(parameter.data)

    def step(self):
        self.t += 1

        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            parameter.grad = np.clip(parameter.grad, -1.0, 1.0)

            m_prev = self.m[parameter]
            v_prev = self.v[parameter]

            grad = parameter.grad

            m_new = m_prev * 0.9 + grad * (1 - 0.9)
            v_new = v_prev * 0.999 + grad**2 * (1 - 0.999)

            m_hat = m_new / (1 - 0.9 ** self.t)
            v_hat = v_new / (1 - 0.999 ** self.t)

            parameter.data -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + 10 ** -8))

            self.m[parameter] = m_new
            self.v[parameter] = v_new

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.grad)