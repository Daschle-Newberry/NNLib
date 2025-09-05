import numpy as np
from NNLib.losses import LossFunction

class MeanSquaredError(LossFunction):
    def forward(self, yHat : np.ndarray, y : np.ndarray):
        return np.mean((yHat - y) ** 2)
    def backward(self, yHat : np.ndarray, y : np.ndarray):
        return 2 * (yHat - y) / y.size