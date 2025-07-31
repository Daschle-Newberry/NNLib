import numpy as np


class Layer:
    def compile(self, input_dim):
        pass

    def forward(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, output_gradients: np.ndarray):
        raise NotImplementedError


class TrainableLayer(Layer):
    @property
    def weights(self):
        raise NotImplementedError

    @weights.setter
    def weights(self, w: np.ndarray):
        raise NotImplementedError

    @property
    def biases(self):
        raise NotImplementedError

    @biases.setter
    def biases(self, b: np.ndarray):
        raise NotImplementedError

    @property
    def w_gradients(self):
        raise NotImplementedError

    @property
    def b_gradients(self):
        raise NotImplementedError

    @property
    def output_dim(self):
        raise NotImplementedError


class UntrainableLayer(Layer):
    @property
    def weights(self):
        raise AttributeError("Untrainable layer has no weights")

    @weights.setter
    def weights(self, w: np.ndarray):
        raise AttributeError("Cannot set weights of an untrainable layer")

    @property
    def biases(self):
        raise AttributeError("Untrainable layer has no biases")

    @biases.setter
    def biases(self, b: np.ndarray):
        raise AttributeError("Cannot set biases of an untrainable layer")

    @property
    def w_gradients(self):
        raise AttributeError("Untrainable layer has no weight gradients")

    @property
    def b_gradients(self):
        raise AttributeError("Untrainable layer has no bias gradients")

    @property
    def output_dim(self):
        raise NotImplementedError
