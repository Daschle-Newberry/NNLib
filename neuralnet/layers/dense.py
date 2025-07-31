import numpy as np

from neuralnet.layers.layer import TrainableLayer


class Dense(TrainableLayer):
    def __init__(self, units: int, name: str):
        self.name = name

        self._units = units

        self._weights = None
        self._biases = None

        self._dW = None
        self._dB = None

        self._input = None
        self._output = None

    def compile(self, input_dim: tuple):
        if len(input_dim) != 1:
            raise ValueError(f"Input of shape {input_dim} cannot be fed to a Dense layer")
        self._weights = np.random.rand(input_dim[0], self._units) * .01
        self._biases = np.random.rand(self._units)
        self._dW = np.zeros_like(self._weights)
        self._dB = np.zeros_like(self._biases)

    def forward(self, x: np.ndarray):
        self._input = x
        self._output = np.dot(x, self._weights) + self._biases
        return self._output

    def backward(self, output_gradients: np.ndarray):
        self._dW += self._input.reshape(-1, 1) @ output_gradients.reshape(1, -1)

        self._dB += output_gradients

        input_gradients = self._weights @ output_gradients
        return input_gradients

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self._weights[:] = w

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, b: np.ndarray):
        self._biases[:] = b

    @property
    def w_gradients(self):
        return self._dW

    @property
    def b_gradients(self):
        return self._dB

    @property
    def output_dim(self):
        return (self._units,)
