from engine.module import Module
from engine.tensor import Tensor
import numpy as np

class ChebLayer(Module):
    def __init__(self,in_features,out_features,degree):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.c = Tensor(np.random.randn(1, in_features, out_features, degree) * 0.1)
            
    def parameters(self):
        return [self.c]
    
    def __call__(self,x):
        x = x.tanh()

        batch_size = x.data.shape[0]

        T_0 = Tensor(np.ones_like(x.data))
        T_1 = x

        T_list = [T_0,T_1]

        for i in range(2,self.degree):
            T_list.append(2*x*T_list[i-1] - T_list[i-2])

        T_stacked = Tensor.stack(T_list, axis=-1)
        T_reshaped = T_stacked.reshape((batch_size, self.in_features, 1, self.degree))

        weighted_T = T_reshaped * self.c

        result = weighted_T.sum(axis=-1).sum(axis=1)

        return result