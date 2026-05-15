from engine.layer import RBFLayer

class KAN:
    def __init__(self,layer_sizes=None):
        self.layer_sizes = layer_sizes
        self.layers = []

        if self.layer_sizes is not None:
            for i in range(0,len(self.layer_sizes)-1):
                self.layers.append(RBFLayer(self.layer_sizes[i],self.layer_sizes[i+1]))

    def __call__(self,x):
        current_data = x

        for layer in self.layers:
            current_data = layer(current_data)

        return current_data
    
    def parameters(self):
        params = []

        for layer in self.layers:
            for item in layer.parameters():
                params.append(item)

        return params