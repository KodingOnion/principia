import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.value import Value
from engine.model import KAN
from engine.optim import AdamOptimizer
from pathlib import Path
import random
import time

def circle_generator(num_points):
    inputs = []
    outputs = []

    for i in range(num_points):
        x = Value(random.uniform(-1.0, 1.0))
        y = Value(random.uniform(-1.0,1.0))

        dist = (x**2 + y**2)**(0.5)

        if dist > 0.6:
            output = Value(0)
        else:
            output = Value(1)

        inputs.append([x,y])
        outputs.append([output])

    return (inputs,outputs)

NUM_POINTS = 200
BATCH_SIZE = 16

x,Y = circle_generator(NUM_POINTS)
data = list(zip(x,Y))
model = KAN([2, 12, 1])

EPOCHS = 1000
LEARNING_RATE = 0.01

optimizer = AdamOptimizer(model.parameters(), learning_rate=LEARNING_RATE)

current_epoch = 0
current_loss = 0

print("--- TRAINING STARTED FOR XOR ---")

for current_epoch in range(1, EPOCHS + 1):
    random.shuffle(data)
    epoch_loss_accumulator = 0.0

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        x_batch, Y_batch = zip(*batch)

        batch_loss = Value(0.0)

        for inp, out in zip(x_batch, Y_batch):
            MSE = (model(inp)[0] - out[0])**2
            batch_loss += MSE

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        epoch_loss_accumulator += batch_loss.data



    if current_epoch % 1 == 0 or current_epoch == 1:
        print(f"EPOCH NUM: {current_epoch} - LOSS: {epoch_loss_accumulator}")

model_dir = Path("models")
filename = model_dir / f"donut_model_epoch_{current_epoch}_loss_{current_loss:.4f}_{time.time()}.json"
model.save(filename)