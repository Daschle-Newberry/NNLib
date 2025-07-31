import numpy as np

class LossFunction:
    def forward(self, yHat : np.ndarray, y : np.ndarray):
        raise NotImplementedError("Loss function not implemented yet")
    def backward(self, yHat : np.ndarray, y : np.ndarray):
        raise NotImplementedError("Loss function not implemented yet")


class MeanSquaredError(LossFunction):
    def forward(self, yHat : np.ndarray, y : np.ndarray):
        return np.mean((yHat - y) ** 2)
    def backward(self, yHat : np.ndarray, y : np.ndarray):
        return 2 * (yHat - y) / y.size

class BinaryCrossEntropy(LossFunction):
    def forward(self, yHat: np.ndarray, y: np.ndarray):
        epsilon = 1E-8
        yHat = np.clip(yHat, epsilon, 1 - epsilon)
        return np.mean(-y * np.log(yHat) - (1 - y) * np.log(1 - yHat))
    def backward(self, yHat: np.ndarray, y: np.ndarray):
        epsilon = 1E-8
        yHat = np.clip(yHat, epsilon, 1 - epsilon)
        return (-y / yHat) + ((1 - y) / (1 - yHat))



