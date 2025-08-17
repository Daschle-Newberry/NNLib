import numpy as np

from NNLib.layers.base import Layer, UntrainableLayer


class Sigmoid(UntrainableLayer):
    def __init__(self):
        self._input = None
        self._output = None
        self._output_dim = None

    def compile(self,input_dim):
        self._output_dim = input_dim

    def forward(self, x: np.ndarray):
        self._input = x
        self._output = sigmoid(x)
        return self._output

    def backward(self, output_gradients: np.ndarray):
        return sigmoid_deriv(self._output) * output_gradients

    @property
    def output_dim(self):
        return self._output_dim

def sigmoid(x : np.ndarray):
    out = np.empty_like(x)

    positive = x >= 0
    negative = ~positive

    out[positive] = 1 / (1 + np.exp(-x[positive]))
    out[negative] = np.exp(x[negative]) / (1 + np.exp(x[negative]))

    return out


def sigmoid_deriv(x : np.ndarray):
    return x * (1 - x)
