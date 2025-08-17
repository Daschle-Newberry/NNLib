import numpy as np

from NNLib.layers.base import Layer, UntrainableLayer

class ReLu(UntrainableLayer):
    def __init__(self):
        self._input = None
        self._output = None
        self._output_dim = None
    def compile(self,input_dim):
        self._output_dim = input_dim
    def forward(self, x: np.ndarray):
        self._input = x
        self._output = relu(x)
        return self._output

    def backward(self, output_gradients: np.ndarray):
        return relu_deriv(self._output) * output_gradients

    @property
    def output_dim(self):
        return self._output_dim
def relu(x : float):
    return np.maximum(0,x)

def relu_deriv(x : float):
    return x > 0