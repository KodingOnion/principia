import random
from engine.value import Value

class RBFEdge:
    def __init__(self,mean=None,width=None,amplitude=None):
        if mean is not None:
            self.mean = Value(mean)
        else:
            data = random.uniform(-1,1)

            self.mean = Value(data)

        if width is not None:
            self.width = Value(width)
        else:
            data = random.uniform(0.1,1)

            self.width = Value(data)

        if amplitude is not None:
            self.amplitude = Value(amplitude)
        else:
            data = random.uniform(-1,1)

            self.amplitude = Value(data)

    def __call__(self, x):
        if not isinstance(x, Value):
            raise TypeError("Incorrect type passed to class")

        data = self.amplitude * ((-self.width)*(x-self.mean)**2).exp()

        return data
