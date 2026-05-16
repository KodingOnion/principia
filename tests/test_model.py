import unittest
import os
import json
from engine.value import Value
from engine.model import KAN

class TestKANModel(unittest.TestCase):

    def test_model_initialization(self):
        """Test if the layer sizes correctly translate into a stack of RBFLayers."""
        # A network with 3 inputs, two hidden layers (4 and 4 nodes), and 2 outputs
        model = KAN([3, 4, 4, 2])
        
        self.assertEqual(len(model.layers), 3, "Should have created exactly 3 layers")
        
        # Check dimensions of the first layer (3 in, 4 out)
        self.assertEqual(model.layers[0].nin, 3)
        self.assertEqual(model.layers[0].nout, 4)
        
        # Check dimensions of the final layer (4 in, 2 out)
        self.assertEqual(model.layers[-1].nin, 4)
        self.assertEqual(model.layers[-1].nout, 2)

    def test_forward_pass_cascade(self):
        """Test if the data successfully passes through the entire stack."""
        model = KAN([2, 5, 1])
        x = [Value(1.0), Value(-1.0)]
        
        out = model(x) # Trigger __call__
        
        self.assertEqual(type(out), list, "Model must return a list of outputs")
        self.assertEqual(len(out), 1, "Model should return exactly 1 output Value based on dimensions")
        self.assertEqual(type(out[0]).__name__, "Value", "Output must be a Value object")

    def test_master_parameter_harvesting(self):
        """Test if the model successfully collects parameters from every layer."""
        model = KAN([2, 3, 2])
        
        # Layer 1: 2 in, 3 out -> 6 edges * 3 params per edge = 18 params
        # Layer 2: 3 in, 2 out -> 6 edges * 3 params per edge = 18 params
        # Total parameters should be 36.
        
        params = model.parameters()
        
        self.assertEqual(len(params), 36, "Model failed to collect the correct total number of parameters")
        self.assertEqual(type(params[0]).__name__, "Value", "Parameters must be naked Value objects")

    def test_full_network_backpropagation(self):
        """The Ultimate Test: Does the chain rule survive the entire deep stack?"""
        model = KAN([2, 4, 1])
        x = [Value(0.5), Value(-0.5)]
        
        # 1. Forward pass
        out = model(x)
        loss = out[0] ** 2
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Check if gradients successfully flowed backwards through the 2D matrices
        params = model.parameters()
        for p in params:
            self.assertNotEqual(p.gradient, 0.0, "Gradient died before reaching a network parameter")
            
        self.assertNotEqual(x[0].gradient, 0.0, "Gradient did not reach network inputs")

class TestModelSerialization(unittest.TestCase):

    def setUp(self):
        """This runs BEFORE every test. We define a temporary filename."""
        self.test_filename = "test_temp_model_checkpoint.json"

    def tearDown(self):
        """This runs AFTER every test. We delete the file so the repo stays clean."""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_save_format_and_dictionary(self):
        """Test if the save method creates a JSON file with the correct architecture and weight lists."""
        model = KAN([2, 3, 1])
        model.save(self.test_filename)
        
        # 1. Did the file physically save to the disk?
        self.assertTrue(os.path.exists(self.test_filename), "Save method failed to create a file.")
        
        # 2. Let's open it manually and check the dictionary structure
        with open(self.test_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertIn("architecture", data, "JSON missing 'architecture' key")
        self.assertIn("weights", data, "JSON missing 'weights' key")
        self.assertEqual(data["architecture"], [2, 3, 1], "Saved architecture blueprint is incorrect")
        
        # 3. Verify total parameters. 
        # Layer 1: 2*3 = 6 edges * 3 params = 18. 
        # Layer 2: 3*1 = 3 edges * 3 params = 9. 
        # Total = 27 floats.
        self.assertEqual(len(data["weights"]), 27, "JSON saved the wrong number of weights")

    def test_load_factory_method(self):
        """Test if the classmethod successfully builds a new model and injects the exact weights."""
        original_model = KAN([2, 2, 1])
        
        # Manually sabotage a specific weight to ensure we aren't just loading default random initialization
        original_model.parameters()[0].data = 99.99
        original_model.save(self.test_filename)
        
        # Use the factory method to build a brand new model from the file
        loaded_model = KAN.load(self.test_filename)
        
        self.assertEqual(type(loaded_model).__name__, "KAN", "Load method did not return a KAN instance")
        self.assertEqual(loaded_model.layer_sizes, [2, 2, 1], "Loaded model built the wrong architecture")
        self.assertEqual(loaded_model.parameters()[0].data, 99.99, "Loaded model failed to inject the saved weights")

    def test_inference_consistency(self):
        """The Ultimate Test: Does a loaded model calculate the exact same math as the original?"""
        model = KAN([3, 4, 2])
        x = [Value(0.5), Value(-0.5), Value(1.0)]
        
        # Get the prediction from the original model
        original_prediction = model(x)[0].data
        
        model.save(self.test_filename)
        
        # Load a clone and get its prediction
        cloned_model = KAN.load(self.test_filename)
        cloned_prediction = cloned_model(x)[0].data
        
        # If the serialization is perfect, the math must be identical down to the decimal
        self.assertEqual(original_prediction, cloned_prediction, "The cloned model's math diverged from the original!")

if __name__ == '__main__':
    unittest.main()