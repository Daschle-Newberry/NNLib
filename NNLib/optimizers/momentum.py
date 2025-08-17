from NNLib.optimizers.base import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float):
        self.learning_rate = learning_rate
        self.beta = beta

        self.weights = None
        self.biases = None
        self.dW = None
        self.dB = None
        self.w_velocities = None
        self.b_velocities = None

    def compile(self, params: dict):
        self.weights = params.get("w")
        self.biases = params.get("b")

        self.dW = params.get("dW")
        self.dB = params.get("dB")

        self.w_velocities = [np.zeros_like(w) for w in self.weights]
        self.b_velocities = [np.zeros_like(b) for b in self.biases]

    def step(self, batch_size: int):
        for i in range(len(self.weights)):
            self.dW[i] /= batch_size
            self.dB[i] /= batch_size

            self.w_velocities[i] *= self.beta
            self.w_velocities[i] += (1- self.beta) * self.dW[i]

            self.b_velocities[i] *= self.beta
            self.b_velocities[i] += (1 - self.beta) * self.dB[i]

            self.weights[i] -= self.learning_rate * self.w_velocities[i]
            self.biases[i] -= self.learning_rate * self.b_velocities[i]

            self.dW[i].fill(0)
            self.dB[i].fill(0)