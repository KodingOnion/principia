from engine.rbf_edge import RBFEdge
from engine.value import Value


class RBFLayer:
    """Layer of Gaussian RBF edges mapping ``nin`` inputs to ``nout`` outputs."""

    def __init__(self,nin=None,nout=None):
        """Build an ``nout x nin`` matrix of ``RBFEdge`` objects."""
        self.nin = nin
        self.nout = nout
        self.out = [[RBFEdge() for _ in range(self.nin)] for _ in range(self.nout)]

    def __call__(self,x):
        """Run a forward pass and return one ``Value`` per output unit."""
        outputs = []

        for row in self.out:
            total = Value(0)
            for edge, xj in zip(row, x):
                total += edge(xj)

            outputs.append(total)

        return outputs

    def parameters(self):
        """Return all learnable edge parameters in a flat list."""
        params = []

        for row in self.out:
            for edge in row:
                params.append(edge.mean)
                params.append(edge.width)
                params.append(edge.amplitude)

        return params