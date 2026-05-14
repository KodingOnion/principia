class Value:
    def __init__(self,data=None,gradient=0.0,children=None,operation=None,backward=None):
        self.data = data # REAL
        self.gradient = gradient # REAL
        self.children = children # SET OF Value
        self.operation = operation # CHAR
        self.backward = backward # FUNCTION

    def __add__(self,b):
        if isinstance(b, int) or isinstance(b,float):
            data = self.data + b
        else:
            data = self.data + b.data

        c = Value(
            data, # Value adding
            0.0, # Gradient stuff
            [self,b], # Children
            '+', # Operation
            None # Backward
        )

        def _backward_recipe():
            self.gradient += (
                self.gradient + (1.0*c.gradient)
                )
            
            b.gradient += (
                b.gradient + (1.0*c.gradient)
            )
    
        c.backward = _backward_recipe

        return c

    def __mul__(self,b):
        c = Value(
            self.data * b.data, # Value adding
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