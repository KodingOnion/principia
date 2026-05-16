"""Train an RBF-KAN model on XOR and save the resulting checkpoint."""

from engine.value import Value
from engine.model import KAN
from pathlib import Path
from engine.optim import AdamOptimizer
import time

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

EPOCHS = 1000
LEARNING_RATE = 0.01

optimizer = AdamOptimizer(model.parameters(), learning_rate=LEARNING_RATE)

current_epoch = 0
current_loss = 0

print("--- TRAINING STARTED FOR XOR ---")

for current_epoch in range(1, EPOCHS + 1):
    total_loss = Value(0.0)

    # 1. Forward Pass (Calculate the loss)
    for inp, out in zip(inputs, outputs):
        MSE = (model(inp)[0] - out[0])**2
        total_loss += MSE

    # 2. Reset the gradients
    optimizer.zero_grad()

    # 3. Backward Pass (Calculate the math)
    total_loss.backward()

    # 4. Take the Adam Step (Update the weights)
    optimizer.step()

    current_loss = total_loss.data

    if current_epoch % 100 == 0 or current_epoch == 1:
        print(f"EPOCH NUM: {current_epoch} - LOSS: {total_loss.data}")

model_dir = Path("models")
filename = model_dir / f"XOR_model_epoch_{current_epoch}_loss_{current_loss:.4f}_{time.time()}.json"
model.save(filename)

print()
print("--- INFERENCE TEST ---")
test_input = [Value(0.0), Value(1.0)]

prediction = model(test_input)[0]

print(f"Raw Output: {prediction.data}")

FINAL_ANSWER = 1 if prediction.data >= 0.5 else 0
print(f"Network predicts: {FINAL_ANSWER}")

