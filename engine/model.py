from engine.layer import RBFLayer
import json
from pathlib import Path
class KAN:
    def __init__(self,layer_sizes=None):
        self.layer_sizes = layer_sizes
        self.layers = []

        if self.layer_sizes is not None:
            for nin, nout in zip(self.layer_sizes, self.layer_sizes[1:]):
                self.layers.append(RBFLayer(nin, nout))

    def __call__(self,x):
        current_data = x

        for layer in self.layers:
            current_data = layer(current_data)

        return current_data
    
    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
    def save(self,filename):
        params = self.parameters()

        dataParams = [param.data for param in params]

        dictionary = {"architecture" : self.layer_sizes, "weights" : dataParams}

        output_path = Path(filename)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(dictionary, file)

    @classmethod
    def load(cls,filepath):
        saved_data = []

        with open(filepath, "r") as file:
            saved_data = json.load(file)

        architecture_list = saved_data["architecture"]
        weights_list = saved_data["weights"]

        model = cls(architecture_list)

        params = model.parameters()
        for param,weight in zip(params,weights_list):
            param.data = weight

        return model
