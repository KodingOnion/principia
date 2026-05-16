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

EPOCHS = 1000
LEARNING_RATE = 0.01

current_epoch = 0
current_loss = 0

for i in range(0,EPOCHS):
    total_loss = Value(0.0)

    for j in range(0,len(inputs)):
        MSE = (model(inputs[j])[0] - outputs[j][0])**2
        total_loss += MSE

    params = model.parameters()

    for parameter in params:
        parameter.gradient = 0.0

    total_loss.backward()

    params = model.parameters()

    for parameter in params:
        parameter.data -= parameter.gradient * LEARNING_RATE

    current_epoch = i+1
    current_loss = total_loss.data

    print(f"EPOCH NUM: {i+1} - LOSS: {total_loss.data}")

filename = f"models\model_epoch_{current_epoch}_loss_{current_loss:.4f}.json"
model.save(filename)

print("\n--- INFERENCE TEST ---")
# Let's test the input [0, 1] - We expect an output close to 1.0
test_input = [Value(0.0), Value(1.0)]

# Run the forward pass
prediction = model(test_input)[0]

print(f"Raw Output: {prediction.data}")

# Since it outputs a raw float, we threshold it at 0.5 to get a clean 0 or 1
final_answer = 1 if prediction.data >= 0.5 else 0
print(f"Network predicts: {final_answer}")