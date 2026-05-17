"""Load a saved KAN model and run simple inference examples."""

from engine.model import KAN
from engine.value import Value

print("\n--- LOADING MODEL ---")
model = KAN.load(r"Z:\principia\models\model_epoch_1000_loss_0.0000_1778949609.3743508.json")
print("Model loaded successfully.")

print("\n--- INFERENCE TEST ---")

test_input = [Value(0.0), Value(1.0)]

prediction = model(test_input)[0]

print(f"Raw Output: {prediction.data}")


final_answer = 1 if prediction.data >= 0.5 else 0
print(f"Network predicts: {final_answer}")


test_input = [Value(0.0), Value(0.0)]

prediction = model(test_input)[0]

print(f"Raw Output: {prediction.data}")


final_answer = 1 if prediction.data >= 0.5 else 0
print(f"Network predicts: {final_answer}")