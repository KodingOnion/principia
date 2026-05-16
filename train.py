from engine.value import Value
from engine.model import KAN
from pathlib import Path

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

current_epoch = 0
current_loss = 0

for current_epoch in range(1, EPOCHS + 1):
    total_loss = Value(0.0)

    for inp, out in zip(inputs, outputs):
        MSE = (model(inp)[0] - out[0])**2
        total_loss += MSE

    params = model.parameters()

    for parameter in params:
        parameter.gradient = 0.0

    total_loss.backward()

    params = model.parameters()

    for parameter in params:
        parameter.data -= parameter.gradient * LEARNING_RATE

    current_loss = total_loss.data

    print(f"EPOCH NUM: {current_epoch} - LOSS: {total_loss.data}")

model_dir = Path("models")
filename = model_dir / f"model_epoch_{current_epoch}_loss_{current_loss:.4f}.json"
model.save(filename)

print()
print("--- INFERENCE TEST ---")
# Let's test the input [0, 1] - We expect an output close to 1.0
test_input = [Value(0.0), Value(1.0)]

# Run the forward pass
prediction = model(test_input)[0]

print(f"Raw Output: {prediction.data}")

# Since it outputs a raw float, we threshold it at 0.5 to get a clean 0 or 1
FINAL_ANSWER = 1 if prediction.data >= 0.5 else 0
print(f"Network predicts: {FINAL_ANSWER}")

