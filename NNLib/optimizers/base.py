from abc import abstractmethod, ABC
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def compile(self, weights: list, biases: list):
        pass

    @abstractmethod
    def step(self, gradient_w: np.ndarray, gradient_b: np.ndarray):
        pass
