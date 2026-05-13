class Value:
    def __init__(self,data=None,gradient=None,children=None,operation=None,backward=None):
        self.__data = data # REAL
        self.__gradient = gradient # REAL
        self.__children = children # SET OF Value
        self.__operation = operation # CHAR
        self.__backward = backward # FUNCTION

    def GetData(self):
        return self.__data
    
    def GetGradient(self):
        return self.__gradient
    
    def GetChildren(self):
        return self.__children
    
    def GetOperation(self):
        return self.__operation
    
    def GetBackward(self):
        return self.__backward
    
    def SetData(self,newData):
        self.__data = newData
    
    def SetGradient(self,newGradient):
        self.__gradient = newGradient

    def SetChildren(self,newChildren):
        self.__children = newChildren

    def SetOperation(self,newOperation):
        self.__operation = newOperation

    def SetBackward(self,newBackward):
        self.__backward = newBackward

    def __add__(self,b):
        c = Value(
            self.GetData() + b.GetData(), # Value adding
            0.0, # Gradient stuff
            [self,b], # Children
            '+', # Operation
            None
        )

        def _backward_recipe(self):
            self.SetGradient(
                self.GetGradient + (1.0*c.GetGradient)
                )
            
            b.SetGradient(
                b.GetGradient + (1.0*c.GetGradient)
            )
    
        c.SetBackward(_backward_recipe)

        return c

    def __mul__(self,b):
        return Value(
            self.GetData() * b.GetData(), # Value adding
            None, # Gradient stuff
            [self,b], # Children
            '*', # Operation
            None
        )