import math


class Value:
    """Scalar value node used to build and differentiate computation graphs.

    Each ``Value`` instance stores data, gradient state, graph connectivity
    (children and operation), and a local backward function used during
    reverse-mode automatic differentiation.
    """

    def __init__(self,data=None,gradient=0.0,children=None,operation=None,backward=None):
        """Initialize a scalar node in the autograd graph."""
        if children is None:
            children = []
        self.data = data
        self.gradient = gradient
        self.children = children
        self.operation = operation
        self._backward = lambda: None

    def __add__(self,b):
        """Return ``self + b`` and attach its local backward rule."""
        if isinstance(b, int) or isinstance(b, float):
            b = Value(b)

        data = self.data + b.data

        c = Value(
            data,
            0.0,
            [self,b],
            '+',
            None
        )

        def _backward_recipe():
            self.gradient += (1.0*c.gradient)
            
            b.gradient += (1.0*c.gradient)
    
        c._backward = _backward_recipe

        return c

    def __mul__(self,b):
        """Return ``self * b`` and attach its local backward rule."""
        if isinstance(b, int) or isinstance(b, float):
            b = Value(b)

        data = self.data * b.data

        c = Value(
            data,
            0.0,
            [self,b],
            '*',
            None
        )

        def _backward_receipe():
            self.gradient += (
                b.data * c.gradient
            )

            b.gradient += (
                self.data * c.gradient
            )
        
        c._backward = _backward_receipe

        return c
    
    def __pow__(self,b):
        """Return ``self ** b`` for numeric exponents with backward rule."""
        if isinstance(b,int) or isinstance(b,float):
            c = Value(
                self.data ** b,
                0.0,
                [self],
                "**",
                None
            )

            def _backward_recipe():
                self.gradient += (
                    b * (self.data ** (b - 1)) * c.gradient
                )
        
            c._backward = _backward_recipe

            return c
        else:
            raise TypeError("Invalid data type for __pow__ in Value class")

    
    def __neg__(self):
        """Return the negated value node."""
        return self * -1.0

    def __sub__(self,b):
        """Return ``self - b`` using addition and negation."""
        return self + (-b)
    
    def __truediv__(self, b):
        """Return ``self / b`` using multiplication by inverse."""
        return self * (b ** -1)
    
    def exp(self):
        """Return ``e ** self`` and attach its local backward rule."""
        c = Value(
            math.exp(self.data),
            0.0,
            [self],
            "exp",
            None
        )

        def _backward_recipe():
            self.gradient += (
                c.data * c.gradient
            )
        
        c._backward = _backward_recipe

        return c
    
    def backward(self):
        """Run reverse-mode autodiff from this node through the full graph."""
        visited = set()
        ordered = []

        def helper(node):
            if node not in visited:
                visited.add(node)

                for child in node.children:
                    helper(child)

                ordered.append(node)

        helper(self)

        if self.gradient is None or self.gradient == 0.0:
            self.gradient = 1.0

        # Reverse topological traversal ensures each node receives all
        # downstream gradients before propagating to its children.
        for node in reversed(ordered):
            node._backward()