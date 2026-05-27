import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.KANLayer import KANLayer
from engine.tensor import Tensor
from engine.mse import mse_loss
import numpy as np

BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 2000

x = np.linspace(-np.pi,np.pi, 100)
x = x.reshape(x.shape[0],1)
Y = np.sin(x)

x = Tensor(x)
Y = Tensor(Y)

model = KANLayer(1,1,10)

for i in range(1,EPOCHS+1):
    predictions = model(x)
    
    MSE = mse_loss(predictions,Y)

    model.zero_grad()

    MSE.backward()

    for param in model.parameters():
        param.data -= param.grad * LEARNING_RATE

    if i % 100 == 0 or i == 1:
        print(f"EPOCH: {i} LOSS: {MSE.data}")