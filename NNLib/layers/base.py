from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    @abstractmethod
    def compile(self, input_dim):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, output_gradients: np.ndarray):
        pass

    @property
    @abstractmethod
    def output_dim(self):
        pass


class TrainableLayer(Layer,ABC):
    @property
    @abstractmethod
    def weights(self):
        pass

    @weights.setter
    @abstractmethod
    def weights(self, w: np.ndarray):
        pass

    @property
    @abstractmethod
    def biases(self):
        pass

    @biases.setter
    @abstractmethod
    def biases(self, b: np.ndarray):
        pass

    @property
    @abstractmethod
    def w_gradients(self):
        pass

    @property
    @abstractmethod
    def b_gradients(self):
        pass



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
