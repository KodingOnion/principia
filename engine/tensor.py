import numpy as np

class Tensor:
    def __init__(self, data=None, children=None, op=None, requires_grad=None):
        self.data = np.array(data, dtype="float32") if data is not None else np.array([], dtype="float32")
        self.grad = np.zeros_like(self.data, dtype="float32")
        self._prev = set(children) if children else set()
        self._op = op if op else ""
        self._requires_grad = requires_grad if requires_grad is not None else True
        self._backward = lambda: None

    def __add__(self, other):
        if isinstance(other,int) or isinstance(other,float):
            other = Tensor(other)

        result = Tensor(
            data=(self.data + other.data),
            children=[self,other],
            op="+"
        )

        def _grad_helper(gradient,target_shape):
            while len(gradient.shape) > len(target_shape):
                gradient = gradient.sum(axis=0)
            
            for i in range(0,len(target_shape)):
                if target_shape[i] == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)

            return gradient

        def _backward():
            self.grad += _grad_helper(result.grad,self.grad.shape)
            other.grad += _grad_helper(result.grad,other.grad.shape)

        result._backward = _backward

        return result

