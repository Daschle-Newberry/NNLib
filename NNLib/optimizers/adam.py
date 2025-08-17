from NNLib.optimizers.base import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate: float, beta1: float, beta2 : float):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_step = 0

        self.weights = None
        self.biases = None
        self.dW = None
        self.dB = None
        self.w_mean = None
        self.b_mean = None

        self.w_variance = None
        self.b_variance = None

    def compile(self, params: dict):
        self.weights = params.get("w")
        self.biases = params.get("b")

        self.dW = params.get("dW")
        self.dB = params.get("dB")

        self.w_mean = [np.zeros_like(w) for w in self.weights]
        self.b_mean = [np.zeros_like(b) for b in self.biases]

        self.w_variance = [np.zeros_like(w) for w in self.weights]
        self.b_variance = [np.zeros_like(b) for b in self.biases]

    def step(self, batch_size: int):
        e = 1e-9
        self.time_step += 1

        for i in range(len(self.weights)):
            self.dW[i] /= batch_size
            self.dB[i] /= batch_size

            self.w_mean[i] = self.beta1 * self.w_mean[i] + (1 - self.beta1) * self.dW[i]
            self.b_mean[i] = self.beta1 * self.b_mean[i] + (1 - self.beta1) * self.dB[i]

            self.w_variance[i] = self.beta2 * self.w_variance[i] + (1 - self.beta2) * (self.dW[i] ** 2)
            self.b_variance[i] = self.beta2 * self.b_variance[i] + (1 - self.beta2) * (self.dB[i] ** 2)

            minus_b1_sq = 1 - (self.beta1 ** self.time_step)
            minus_b2_sq = 1 - (self.beta2 ** self.time_step)

            w_mean_corrected = self.w_mean[i] / minus_b1_sq
            b_mean_corrected = self.b_mean[i] / minus_b1_sq

            w_variance_corrected = self.w_variance[i] / minus_b2_sq
            b_variance_corrected = self.b_variance[i] / minus_b2_sq

            self.weights[i] -= self.learning_rate * (w_mean_corrected / (np.sqrt(w_variance_corrected) + e))
            self.biases[i] -= self.learning_rate * (b_mean_corrected / (np.sqrt(b_variance_corrected) + e))

            self.dW[i].fill(0)
            self.dB[i].fill(0)