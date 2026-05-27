import numpy as np

class Tensor:
    def __init__(self, data=None, children=None, op=None, requires_grad=None):
        self.data = np.array(data, dtype="float32") if data is not None else np.array([], dtype="float32")
        self.grad = np.zeros_like(self.data, dtype="float32")
        self._prev = set(children) if children else set()
        self._op = op if op else ""
        self._requires_grad = requires_grad if requires_grad is not None else True
        self._backward = lambda: None

    def _unbroadcast(self, gradient,target_shape):
            while len(gradient.shape) > len(target_shape):
                gradient = gradient.sum(axis=0)
            
            for i in range(0,len(target_shape)):
                if target_shape[i] == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)

            return gradient

    def __add__(self, other):
        if isinstance(other,int) or isinstance(other,float):
            other = Tensor(other)

        result = Tensor(
            data=(self.data + other.data),
            children=[self,other],
            op="+"
        )

        def _backward():
            self.grad += self._unbroadcast(result.grad,self.grad.shape)
            other.grad += self._unbroadcast(result.grad,other.grad.shape)

        result._backward = _backward

        return result

    def __pow__(self,exponent):
        if isinstance(exponent,int) or isinstance(exponent,float):
            result = Tensor(
                data=(self.data ** exponent),
                children=[self],
                op='**'
            )

            def _backward():
                self.grad += result.grad * (exponent * (self.data ** (exponent-1)))

            result._backward = _backward

            return result

        else:
            raise TypeError("Wrong type passed to __pow__")
        
    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            other = Tensor(other)

        result = Tensor(
            data=(self.data*other.data),
            children=[self,other],
            op='*'
        )

        def _backward():
            raw_grad_self = result.grad * other.data
            raw_grad_other = result.grad * self.data

            self.grad += self._unbroadcast(raw_grad_self, self.grad.shape)
            other.grad += self._unbroadcast(raw_grad_other, other.grad.shape)

        result._backward = _backward

        return result
    
    def __neg__(self):
        return self * -1.0
    
    def __sub__(self,other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other**-1.0)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def exp(self):
        result = Tensor(
            data=np.exp(self.data),
            children=[self],
            op='exp'
        )

        def _backward():
            self.grad += result.data * result.grad

        result._backward = _backward

        return result
    
    def backward(self):
        visited = set()
        ordered = []

        def helper(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    helper(child)

                ordered.append(node)

        helper(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(ordered):
            node._backward()

    def __matmul__(self, other):
        result = Tensor(
            data=(self.data @ other.data),
            children=[self,other],
            op='@'
        )

        def _backward():
            raw_grad_self = result.grad @ other.data.T
            raw_grad_other = self.data.T @ result.grad

            self.grad += self._unbroadcast(raw_grad_self, self.grad.shape)
            other.grad += self._unbroadcast(raw_grad_other, other.grad.shape)

        result._backward = _backward

        return result
    
    def sum(self):
        result = Tensor(
            data=np.sum(self.data),
            children=[self],
            op='sum'
        )

        def _backward():
            self.grad += np.ones_like(self.data) * result.grad

        result._backward = _backward

        return result
    
    def mse_loss(self, predictions, targets):
        """
        Calculates the Mean Squared Error between two Tensors.
        """
        # 1. Calculate the squared differences
        differences = predictions - targets
        squared_differences = differences ** 2
        
        # 2. Sum them all up (using your new method!)
        total_loss = squared_differences.sum()
        
        # 3. Divide by the total number of elements to get the mean
        # (We use targets.data.size to get the total number of items in the numpy array)
        n = targets.data.size
        mean_loss = total_loss * (1.0 / n)
        
        return mean_loss