from abc import ABC, abstractmethod

import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def forward(self, yHat : np.ndarray, y : np.ndarray):
        pass
    @abstractmethod
    def backward(self, yHat : np.ndarray, y : np.ndarray):
        pass

