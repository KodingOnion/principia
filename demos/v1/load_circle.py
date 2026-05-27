"""Load a saved KAN model and verify circular decision boundaries."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.v1.model import KAN
from engine.v1.value import Value

# 1. Load the model. Update the string below to point to your specific 
# .json checkpoint in the models/ directory.
MODEL_PATH = "Z:\principia\models\donut_model_epoch_1000_loss_0.0000_1779006611.9611297.json"

print(f"\n--- LOADING MODEL: {MODEL_PATH} ---")
model = KAN.load(MODEL_PATH)
print("Model loaded successfully.")

# 2. Define test cases.
# Radius was 0.6 in circle.py.
test_points = [
    {"coord": [0.0, 0.0],   "desc": "Bullseye (Inside)"},
    {"coord": [0.4, 0.4],   "desc": "Deep Inside (dist ~0.56)"},
    {"coord": [0.4, 0.45],  "desc": "On the edge (dist ~0.60)"},
    {"coord": [0.7, 0.7],   "desc": "Outside (dist ~0.99)"},
    {"coord": [1.0, 0.0],   "desc": "Far Outside"}
]

print("\n--- INFERENCE TESTS ---")

for test in test_points:
    # Convert raw floats to Value objects for the engine
    inputs = [Value(v) for v in test["coord"]]
    
    # Run the forward pass
    # model() returns a list of outputs; we take the first (and only) index [0]
    prediction = model(inputs)[0]
    
    # Apply the threshold logic
    final_label = 1 if prediction.data >= 0.5 else 0
    
    print(f"{test['desc']} {test['coord']}:")
    print(f"  Raw Output: {prediction.data:.4f}")
    print(f"  Network Predicts: {final_label}")
    print("-" * 30)