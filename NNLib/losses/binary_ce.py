import numpy as np
from NNLib.losses import LossFunction

class BinaryCrossEntropy(LossFunction):
    def forward(self, yHat: np.ndarray, y: np.ndarray):
        epsilon = 1E-8
        yHat = np.clip(yHat, epsilon, 1 - epsilon)
        return np.mean(-y * np.log(yHat) - (1 - y) * np.log(1 - yHat))
    def backward(self, yHat: np.ndarray, y: np.ndarray):
        epsilon = 1E-8

        n = len(yHat[0])
        yHat = np.clip(yHat, epsilon, 1 - epsilon)
        res = (-y / yHat) + ((1 - y) / (1 - yHat))

        return res / n