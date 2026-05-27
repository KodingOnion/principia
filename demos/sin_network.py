import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.KAN import KAN
from engine.tensor import Tensor
from engine.mse import mse_loss
from engine.adam_optim import AdamOptimiser
import numpy as np

LEARNING_RATE = 0.01
EPOCHS = 2500

X_data = np.linspace(-np.pi, np.pi, 100).reshape(100, 1)
Y_data = np.sin(X_data)
X = Tensor(X_data)
Y = Tensor(Y_data)

model = KAN(layer_sizes=[1, 5, 1], num_centers=10)

optimiser = AdamOptimiser(model.parameters(), learning_rate=LEARNING_RATE)

for i in range(1,EPOCHS+1):
    predictions = model(X)

    MSE = mse_loss(predictions,Y)

    model.zero_grad()

    MSE.backward()

    optimiser.step()

    if i % 100 == 0 or i == 1:
        print(f"EPOCH NUM:{i} LOSS:{MSE.data}")

import matplotlib.pyplot as plt  # type: ignore

predictions = model(X).data

plt.plot(X_data, Y_data, label="True Sine Wave", color="blue", linewidth=2)
plt.plot(X_data, predictions, label="KAN Approximation", color="red", linestyle="dashed", linewidth=2)
plt.legend()
plt.title(f"KAN Sine Wave Approximation (Final Loss: {MSE.data:.5f})")
plt.show()