from abc import abstractmethod, ABC
import numpy as np

from neuralnet.layer import Layer


class Optimizer(ABC):
    @abstractmethod
    def compile(self, weights : list, biases : list):
        pass
    @abstractmethod
    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray):
        pass


class Basic(Optimizer):
    def __init__(self, learning_rate : float):
        self.learning_rate = learning_rate
        self.weights = None
        self.biases = None


    def compile(self, weights : list, biases : list):
        self.weights = weights
        self.biases = biases

    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray):
            for w, g in zip(self.weights, gradient_w):
                w -= self.learning_rate * g
            for b , g in zip(self.biases, gradient_b):
                b -= self.learning_rate * g

class RMSProp(Optimizer):
    def __init__(self, learning_rate : float, rho : float):
        self.learning_rate = learning_rate
        self.rho = rho

        self.weights = None
        self.biases = None

        self.w_sq_avg = None
        self.b_sq_avg = None

    def compile(self, weights : list, biases : list):
        self.weights = weights
        self.biases = biases

        self.w_sq_avg = [np.zeros_like(w) for w in self.weights]
        self.b_sq_avg = [np.zeros_like(b) for b in self.biases]

    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray):
        e = 1E-8

        for i, (avg, grad) in enumerate(zip(self.w_sq_avg, gradient_w)):
            self.w_sq_avg[i] = self.rho * self.w_sq_avg[i] + (1 - self.rho) * (grad * grad)

        for i, (avg, grad) in enumerate(zip(self.b_sq_avg, gradient_b)):
            self.b_sq_avg[i] = self.rho * self.b_sq_avg[i] + (1 - self.rho) * (grad * grad)


        for avg, w, g in zip(self.w_sq_avg, self.weights, gradient_w):
            w -= (self.learning_rate / (np.sqrt(avg + e))) * g

        for avg, b, g in zip(self.b_sq_avg, self.biases, gradient_b):
            b -= (self.learning_rate / (np.sqrt(avg + e))) * g


class Momentum(Optimizer):
    def __init__(self, learning_rate : float, beta : float):
        self.learning_rate = learning_rate
        self.beta = beta

        self.weights = None
        self.biases = None
        self.w_velocities = None
        self.b_velocities = None

    def compile(self, weights : list, biases : list):
        self.weights = weights
        self.biases = biases

        self.w_velocities = [np.zeros_like(w) for w in self.weights]
        self.b_velocities = [np.zeros_like(b) for b in self.biases]


    def step(self, gradient_w: np.ndarray, gradient_b : np.ndarray):

        for v, g in zip(self.w_velocities, gradient_w):
            v *= self.beta
            v += (1 - self.beta) * g

        for w, v in zip(self.weights, self.w_velocities):
            w -= self.learning_rate * v


        for v, g in zip(self.b_velocities, gradient_b):
            v *= self.beta
            v += (1 - self.beta) * g

        for b, v in zip(self.biases, self.b_velocities):
            b -= self.learning_rate * v