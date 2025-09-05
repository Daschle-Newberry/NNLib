import numpy as np
from NNLib.losses import LossFunction

class SoftMaxCrossEntropy(LossFunction):
    def forward(self, yHat : np.ndarray, y : np.ndarray):
        epsilon = 1E-8

        exponent = np.exp(yHat - np.max(yHat, axis = 1, keepdims = True))
        softmax = exponent / np.sum(exponent, axis = 1, keepdims = True)

        pre_log = np.clip(softmax[np.arange(len(yHat)),y], epsilon, 1 - epsilon)


        loss = -np.log(pre_log)

        return np.mean(loss)

    def backward(self,yHat: np.ndarray, y: np.ndarray):
        exponent = np.exp(yHat - np.max(yHat, axis=1, keepdims=True))
        softmax = exponent / np.sum(exponent, axis=1, keepdims=True)

        softmax[np.arange(len(yHat)),y] -= 1

        return softmax