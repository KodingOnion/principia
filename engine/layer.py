from engine.rbf_edge import RBFEdge
from engine.value import Value

class RBFLayer:
    def __init__(self,nin=None,nout=None):
        self.nin = nin
        self.nout = nout
        self.out = []

        for i in range(0,self.nout):
            self.out.append([])
            for j in range(0,self.nin):
                self.out[i].append(RBFEdge())

    def __call__(self,x):
        outputs = []

        for i in range(0,self.nout):
            total = Value(0)
            for j in range(0,self.nin):
                total += (self.out[i][j](x[j]))

            outputs.append(total)

        return outputs

    def parameters(self):
        params = []

        for i in range(0,self.nout):
            for j in range(0,self.nin):
                params.append(self.out[i][j].mean)
                params.append(self.out[i][j].width)
                params.append(self.out[i][j].amplitude)

        return params