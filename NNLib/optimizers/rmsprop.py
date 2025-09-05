from .base import Optimizer
import numpy as np


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, rho: float):
        self.learning_rate = learning_rate
        self.rho = rho

        self.weights = None
        self.biases = None
        self.dW = None
        self.dB = None
        self.w_sq_avg = None
        self.b_sq_avg = None

    def compile(self, params: dict):
        self.weights = params.get("w")
        self.biases = params.get("b")

        self.dW = params.get("dW")
        self.dB = params.get("dB")
        self.w_sq_avg = [np.zeros_like(w) for w in self.weights]
        self.b_sq_avg = [np.zeros_like(b) for b in self.biases]

    def step(self, batch_size: int):
        e = 1E-8


        for i in range(len(self.weights)):
            self.dW[i] /= batch_size
            self.dB[i] /= batch_size

            self.w_sq_avg[i] = self.rho * self.w_sq_avg[i] + (1 - self.rho) * (self.dW[i] * self.dW[i])
            self.b_sq_avg[i] = self.rho * self.b_sq_avg[i] + (1 - self.rho) * (self.dB[i] * self.dB[i])

            self.weights[i] -= (self.learning_rate / (np.sqrt(self.w_sq_avg[i] + e))) * self.dW[i]
            self.biases[i] -= (self.learning_rate / (np.sqrt(self.b_sq_avg[i] + e))) * self.dB[i]

            self.dW[i].fill(0)
            self.dB[i].fill(0)

