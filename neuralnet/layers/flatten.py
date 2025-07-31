import numpy as np

from neuralnet.layers.layer import UntrainableLayer


class Flatten(UntrainableLayer):
    def __init__(self):
        self._input_dim = None
        self._output_dim = None
        self._input = None
        self._output = None

    def compile(self, input_dim: tuple):
        self._input_dim = input_dim
        self._output_dim = (np.prod(input_dim),)

    def forward(self, x: np.ndarray):
        self._input = x
        self._output = x.reshape(-1)

        if self._output.shape != self._output_dim:
            raise ValueError(
                f"Flattened array of shape {self._output.shape} does not match required shape of {self.output_dim}")
        return self._output

    def backward(self, output_gradients: np.ndarray):
        return output_gradients.reshape(self._input_dim)

    @property
    def output_dim(self):
        return self._output_dim
