from abc import abstractmethod, ABC

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def compile(self, weights: list, biases: list):
        pass

    @abstractmethod
    def step(self, gradient_w: np.ndarray, gradient_b: np.ndarray):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

        self.weights = None
        self.biases = None
        self.dW = None
        self.dB = None

    def compile(self, params: dict):
        self.weights = params.get("w")
        self.biases = params.get("b")

        self.dW = params.get("dW")
        self.dB = params.get("dB")

    def step(self, batch_size: int):

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * (self.dW[i] / batch_size)
            self.biases[i] -= self.learning_rate * (self.dB[i] / batch_size)

            self.dW[i].fill(0)
            self.dB[i].fill(0)


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