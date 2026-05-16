from engine.rbf_edge import RBFEdge
from engine.value import Value

class RBFLayer:
    def __init__(self,nin=None,nout=None):
        self.nin = nin
        self.nout = nout
        self.out = [[RBFEdge() for _ in range(self.nin)] for _ in range(self.nout)]

    def __call__(self,x):
        outputs = []

        for row in self.out:
            total = Value(0)
            for edge, xj in zip(row, x):
                total += edge(xj)

            outputs.append(total)

        return outputs

    def parameters(self):
        params = []

        for row in self.out:
            for edge in row:
                params.append(edge.mean)
                params.append(edge.width)
                params.append(edge.amplitude)

        return params