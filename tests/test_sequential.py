import unittest
from principia import Sequential, Tensor

class DummyLayer:
    def __init__(self, name, param_list, math_operation):
        self.name = name
        self._params = param_list
        self.math_operation = math_operation

    def __call__(self, x):
        # We just apply a simple math operation to track the tensor's journey
        return self.math_operation(x)

    def parameters(self):
        return self._params


class TestSequential(unittest.TestCase):

    def test_sequential_initialisation(self):
        """Test that Sequential correctly stores the list of layers."""
        layer1 = DummyLayer("layer1", [], lambda x: x)
        layer2 = DummyLayer("layer2", [], lambda x: x)
        
        model = Sequential([layer1, layer2])
        
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.layers[0].name, "layer1")
        self.assertEqual(model.layers[1].name, "layer2")

    def test_sequential_forward_pass(self):
        """Test the 'bucket brigade' output passing."""
        # Layer 1 adds 5, Layer 2 multiplies by 2
        layer1 = DummyLayer("layer1", [], lambda x: x + 5)
        layer2 = DummyLayer("layer2", [], lambda x: x * 2)
        
        model = Sequential([layer1, layer2])
        
        # Input is 10. 
        # After layer 1: 10 + 5 = 15
        # After layer 2: 15 * 2 = 30
        result = model(10)
        
        self.assertEqual(result, 30)

    def test_sequential_parameter_harvesting(self):
        """Test that parameters from all layers are flattened into one list."""
        # We use strings here just to prove the list aggregation works
        layer1 = DummyLayer("layer1", ["weight_1", "bias_1"], lambda x: x)
        layer2 = DummyLayer("layer2", ["weight_2"], lambda x: x)
        layer3 = DummyLayer("layer3", [], lambda x: x) # Empty parameter layer
        
        model = Sequential([layer1, layer2, layer3])
        
        params = model.parameters()
        
        self.assertEqual(len(params), 3)
        self.assertEqual(params, ["weight_1", "bias_1", "weight_2"])


if __name__ == '__main__':
    unittest.main()