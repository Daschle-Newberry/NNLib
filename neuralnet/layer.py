import numpy as np

from neuralnet.functions import activations


class Layer:
    def __init__(self, units : int, activation : str):
        self.units = units

        self.w = None
        self.b = None

        self.dW = None
        self.dB = None

        self.activation = activations.get(activation)
        self.activation_deriv = activations.get_deriv(activation)
        self.activation_type = activation


    def compute(self, x : np.ndarray):
        return self.activation(np.dot(x,self.w) + self.b)