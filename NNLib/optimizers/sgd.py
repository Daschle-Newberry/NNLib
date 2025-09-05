from .base import Optimizer

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
