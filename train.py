from engine.value import Value
from engine.model import KAN

model = KAN([2, 5, 1])

inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

outputs = [
    [0],
    [1],
    [1],
    [0]
]

inputs = [[Value(v) for v in row] for row in inputs]
outputs = [[Value(v) for v in row] for row in outputs]

EPOCHS = 500
LEARNING_RATE = 0.01

for i in range(0,EPOCHS):
    total_loss = Value(0.0)

    for i in range(0,len(inputs)):
        MSE = (model(inputs[i])[0] - outputs[i][0])**2
        total_loss += MSE

    params = model.parameters()

    for parameter in params:
        parameter.gradient = 0.0

    total_loss.backward()

    params = model.parameters()

    for parameter in params:
        parameter.data -= parameter.gradient * LEARNING_RATE

    print(f"EPOCH NUM: {i} - LOSS: {total_loss.data}")