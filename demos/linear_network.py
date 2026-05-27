import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.linear import Linear
from engine.tensor import Tensor
from engine.mse import mse_loss
import numpy as np

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 1000

x = np.random.randint(10, 100, size=(BATCH_SIZE, 1)).astype(float)

Y = x * 0.03

x = Tensor(x)
Y = Tensor(Y)

model = Linear(1,1)

for i in range(EPOCHS):
    predictions = model(x)

    MSE = mse_loss(predictions,Y)

    model.zero_grad()

    MSE.backward()

    params = model.parameters()

    for param in params:
        param.data -= param.grad * LEARNING_RATE

    if i % 10 == 0 or i == 0:
        print(f"EPOCH NUM:{i} LOSS:{MSE.data}")

print(model.w.data)