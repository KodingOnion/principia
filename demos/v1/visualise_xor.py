import numpy
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.v1.model import KAN
from engine.v1.value import Value
model = KAN.load(r"Z:\principia\models\XOR_model_epoch_1000_loss_0.0000_1779794052.6080422.json")

xx = numpy.linspace(-0.5,1.5,50)
yy = numpy.linspace(-0.5,1.5,50)

numpy_mesh = numpy.meshgrid(xx,yy)

zz = numpy.zeros_like(numpy_mesh[0])

for i in range(numpy_mesh[0].shape[0]):
    for j in range(numpy_mesh[0].shape[1]):
        model([Value(numpy_mesh[0][i][j]), Value(numpy_mesh[1][i][j])])
        zz[i][j] = model([Value(numpy_mesh[0][i][j]), Value(numpy_mesh[1][i][j])])[0].data

plt.contourf(xx, yy, zz, levels=20)
plt.scatter([0,0,1,1], [0,1,0,1], c='black', marker='x')
plt.title("KAN Decision Boundary - XOR")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.colorbar(label='Output Value')
plt.show()