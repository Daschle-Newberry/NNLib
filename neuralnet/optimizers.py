from abc import abstractmethod, ABC
import numpy as np

from neuralnet.layer import Layer


class Optimizer(ABC):
    @abstractmethod
    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray,  layer_num : int):
        pass


class StandardMomentum(Optimizer):
    def __init__(self,beta : float, layers : list[Layer], learning_rate : float):
        self.beta = beta
        self.learning_rate = learning_rate
        self.w_velocities = [np.zeros_like(layer.w) for layer in layers]
        self.b_velocities = [np.zeros_like(layer.b) for layer in layers]

    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray, layer_num : int):

        return (self.beta * self.w_velocities[layer_num]) + (1 - self.beta) * gradient_w, (self.beta * self.b_velocities[layer_num]) + (1 - self.beta) * gradient_b


