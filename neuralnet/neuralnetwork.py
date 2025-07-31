import time

import numpy as np

from neuralnet.losses import losses
from neuralnet.layers.layer import TrainableLayer
from neuralnet.logs.trainingmonitor import TrainingMonitor
from neuralnet.losses.losses import LossFunction
from neuralnet.optimizers import Optimizer

class NeuralNetwork:
    def __init__(self, layers : list, loss : LossFunction):
        self.layers = layers
        self.loss = loss
        self.optimizer = None
        self.input_size = None

    def __str__(self):
        res = "LAYER ============= NEURONS ============= ACTIVATION\n"
        for i, layer in enumerate(self.layers):
            res += f"{i}" + (6 - len(str(i))) * " " + "             " + f"{layer.units}" + (9 - len(str(layer.units))) * " " + "             " + f"{layer.activation_type}" + (11 - len(str(layer.activation_type))) * " " + "\n"

        return res

    def compile(self, input_size : int, optimizer : Optimizer):
        self.optimizer = optimizer
        self.input_size = input_size
        for i, layer in enumerate(self.layers):
            input_dim = input_size if i == 0 else self.layers[i - 1].output_dim
            layer.compile(input_dim)
        self.optimizer.compile(self.get_params())


    def predict(self, x : np.ndarray):
        a_i = x
        for layer in self.layers:
            a_i = layer.forward(a_i)
        return a_i

    def fit(
            self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            batch_size : int = 32
    ) -> None:
        """
        Trains model using batched gradient descent. Model weights and biases are updated after each batch.

        :param X: Input features, shape (n_samples, n_features)
        :param y: Desired outputs, shape (n_samples, n_outputs)
        :param epochs: Amount of iterations through the training data
        :param batch_size: Number of sampler per training batch
        :return: None
        """


        logger = TrainingMonitor(epochs,len(X) / batch_size)

        for epoch in range(1,epochs + 1):


            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]


            for batch_start in range(0, len(X), batch_size):
                batch_x = X[batch_start : batch_start + batch_size]
                batch_y = y[batch_start : batch_start + batch_size]

                cost = 0


                for i in range(0, len(batch_x)):

                    a = self.forward_pass(batch_x[i])
                    cost  += self.loss.forward(a, batch_y[i])

                    self.backward_pass(a, batch_y[i])

                self.optimizer.step(len(batch_x))
                cost /= len(batch_x)
                logger.update(epoch,np.ceil(batch_start / batch_size), cost)




    def forward_pass(
            self,
            X : np.ndarray
    ) -> np.ndarray:
        """
        Passes through the network to calculate and store the activations of each layer.

        :param X: Input features, shape (n_samples, n_features)
        :return: y (X): Vector of network outputs
        """
        if X.shape != self.input_size:
            raise ValueError(f"Shape mismatch between X and compiled input size, is your training data or input size correct?")
        for layer in self.layers:
            X = layer.forward(X)

        return X


    def backward_pass(
            self,
            yHat : np.ndarray,
            y : np.ndarray
    ) -> None:
        """
        Calculates the gradient of every layer using back propagation.

        :param yHat: List of outputs from the last forward pass
        :param y: Desired outputs, shape (n_samples, n_outputs)
        :return: None
        """

        if yHat.shape != y.shape:
            raise ValueError(f"Shape mismatch between yHat and y, is your training data correct? {yHat.shape},{y.shape}")

        output_gradient = self.loss.backward(yHat, y)

        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def get_params(self):
        return {"w" : [layer.weights for layer in self.layers if isinstance(layer,TrainableLayer)],
                "b" : [layer.biases for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dW" : [layer.w_gradients for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dB" : [layer.b_gradients for layer in self.layers if isinstance(layer,TrainableLayer)]
                }

    def get_layer(self, index : int):
        return self.layers[index]




