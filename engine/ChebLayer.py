from engine.module import Module
from engine.tensor import Tensor
import numpy as np

class ChebLayer(Module):
    def __init__(self,in_features,out_features,degree):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.c_list = []

        for i in range(degree):
            self.c_list.append(
                Tensor(
                    np.random.randn(1,self.in_features,self.out_features)
                    )
                )
            
    def parameters(self):
        return self.c_list
    
    def __call__(self,x):
        x = x.tanh()

        batch_size = x.data.shape[0]

        T_0 = Tensor(np.ones_like(x.data))
        T_1 = x

        T_list = [T_0,T_1]

        for i in range(2,self.degree):
            T_list.append(2*x*T_list[i-1] - T_list[i-2])

        result = Tensor(np.zeros((batch_size, self.out_features)))

        for n,T in enumerate(T_list):
            T_reshaped = T.reshape((batch_size, self.in_features, 1))
            term = T_reshaped * self.c_list[n]
            term = term.sum(axis=1)
            result = result + term

        return result