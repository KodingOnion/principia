import math

class Value:
    def __init__(self,data=None,gradient=0.0,children=None,operation=None,backward=None):
        self.data = data # REAL
        self.gradient = gradient # REAL
        self.children = children # SET OF Value
        self.operation = operation # CHAR
        self.backward = backward # FUNCTION

    def __add__(self,b):
        if isinstance(b, int) or isinstance(b, float):
            b = Value(b)

        data = self.data + b.data

        c = Value(
            data, # Value adding
            0.0, # Gradient stuff
            [self,b], # Children
            '+', # Operation
            None # Backward
        )

        def _backward_recipe():
            self.gradient += (1.0*c.gradient)
            
            b.gradient += (1.0*c.gradient)
    
        c.backward = _backward_recipe

        return c

    def __mul__(self,b):
        if isinstance(b, int) or isinstance(b, float):
            b = Value(b)

        data = self.data * b.data

        c = Value(
            data, # Value adding
            0.0, # Gradient stuff
            [self,b], # Children
            '*', # Operation
            None # Backward
        )

        def _backward_receipe():
            self.gradient += (
                b.data * c.gradient
            )

            b.gradient += (
                self.data * c.gradient
            )
        
        c.backward = _backward_receipe

        return c
    
    def __pow__(self,b):
        if isinstance(b,int) or isinstance(b,float):
            c = Value(
                self.data ** b,
                0.0,
                [self],
                "^",
                None
            )

            def _backward_recipe():
                self.gradient += (
                    b * (self.data ** (b - 1)) * c.gradient
                )
        
            c.backward = _backward_recipe

            return c
        else:
            raise TypeError("Invalid data type for __pow__ in Value class")

    
    def __neg__(self):
        return self * -1.0

    def __sub__(self,b):
        return self + (-b)
    
    def __truediv__(self, b):
        return self * (b ** -1)
    
    def exp(self):
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
        
        c.backward = _backward_recipe

        return c