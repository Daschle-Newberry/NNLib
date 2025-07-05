import numpy as np

import activations
from activations import identity


class Layer:
    def __init__(self, units : int, activation : str):
        self.units = units

        self.w = None
        self.b = None
        self.activation = activations.get(activation)
        self.activation_deriv = activations.get_deriv(activation)

        if self.activation == identity:
            raise Exception(f"Unknown activation function {activation}")
        self.activation_type = activation

    def compute(self, x : np.ndarray):
        return self.activation(np.dot(x,self.w) + self.b)

    def pre_activation(self, x : np.ndarray):
        return np.dot(x,self.w) + self.b

    def post_activation(self, x : np.ndarray):
        return self.activation(x)