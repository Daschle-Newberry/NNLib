
import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.network = layers

    def __str__(self):
        res = "LAYER ============= NEURONS ============= ACTIVATION\n"
        for i, layer in enumerate(self.network):
            res += f"{i}" + (6 - len(str(i))) * " " + "             " + f"{layer.units}" + (9 - len(str(layer.units))) * " " + "             " + f"{layer.activation_type}" + (11 - len(str(layer.activation_type))) * " " + "\n"

        return res

    def compile(self, input_size : int):
        for i, layer in enumerate(self.network):
            input_dim = input_size if i == 0 else self.network[i - 1].units
            layer.w = np.zeros((input_dim, layer.units))
            layer.b = np.zeros(layer.units)

    def predict(self, x : np.ndarray):
        a_i = x
        for layer in self.network:
            a_i = layer.compute(a_i)
        return a_i








