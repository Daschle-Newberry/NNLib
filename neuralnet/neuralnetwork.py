
import numpy as np
import numpy.random

from neuralnet.functions import losses
from neuralnet.layer import Layer
from neuralnet.optimizers import Optimizer


class NeuralNetwork:
    def __init__(self, layers : list[Layer], loss_func : str, optimizer : Optimizer):
        self.layers = layers
        self.loss = losses.get(loss_func)
        self.loss_deriv = losses.get_deriv(loss_func)
        self.optimizer = optimizer
        numpy.random.seed(1234)

    def __str__(self):
        res = "LAYER ============= NEURONS ============= ACTIVATION\n"
        for i, layer in enumerate(self.layers):
            res += f"{i}" + (6 - len(str(i))) * " " + "             " + f"{layer.units}" + (9 - len(str(layer.units))) * " " + "             " + f"{layer.activation_type}" + (11 - len(str(layer.activation_type))) * " " + "\n"

        return res
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.w)
        return weights

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.b)
        return biases


    def compile(self, input_size : int):
        for i, layer in enumerate(self.layers):
            input_dim = input_size if i == 0 else self.layers[i - 1].units
            layer.w = np.random.rand(input_dim, layer.units)
            layer.b = np.random.rand(layer.units)

    def predict(self, x : np.ndarray):
        a_i = x
        for layer in self.layers:
            a_i = layer.compute(a_i)
        return a_i

    def fit_stochastic(
            self,
            X : np.ndarray,
            y : np.ndarray,
            learning_rate : float,
            epochs : int
    ) -> None:
        """

        Trains model using stochastic gradient descent. Model weights and biases are updated after each training sample.

        :param X: Input features, shape (n_samples, n_features)
        :param y: Desired outputs, shape (n_samples, n_outputs)
        :param learning_rate: Scalar for gradients while learning
        :param epochs: Amount of iterations through the training data
        :return: None
        """
        for epoch in range(epochs):
            if epoch % (.1 * epochs) == 0: print(f"\nEpoch : {epoch}")

            #Shuffle the training data
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            cost = 0

            #Loop over all training data and adjust gradients per sample
            for i in range(0, len(X)):
                a = self.forward(X[i])
                cost += self.loss(a[-1], y[i])

                g_w, g_b = self.backward(a, y[i])

                for j, layer in enumerate(self.layers):
                    step_w, step_b = self.optimizer.step(g_w[j],g_b[j],j)

                    layer.w -= learning_rate * step_w
                    layer.b -= learning_rate * step_b



            print(f"\r Cost : {cost / len(X)}", end = '')



    # Epoch: 900
    # Cost: 0.04462793854352141
    def fit_batched(
            self,
            X : np.ndarray,
            y : np.ndarray,
            learning_rate : float,
            epochs : int,
            batch_size : int
    ) -> None:
        """
        Trains model using batched gradient descent. Model weights and biases are updated after each batch.

        :param X: Input features, shape (n_samples, n_features)
        :param y: Desired outputs, shape (n_samples, n_outputs)
        :param learning_rate: Scalar for gradients while learning
        :param epochs: Amount of iterations through the training data
        :param batch_size: Number of sampler per training batch
        :return: None
        """
        for epoch in range(epochs):
            if epoch % (.1 * epochs) == 0: print(f"\nEpoch : {epoch}")

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for batch_start in range(0, len(X), batch_size):
                cost = 0

                w_gradients = [np.zeros_like(layer.w) for layer in self.layers]
                b_gradients = [np.zeros_like(layer.b) for layer in self.layers]


                batch_x = X[batch_start : batch_start + batch_size]
                batch_y = y[batch_start : batch_start + batch_size]


                for i in range(0, len(batch_x)):
                    a = self.forward(batch_x[i])

                    cost += self.loss(a[-1], batch_y[i])

                    g_w, g_b = self.backward(a, y[i])

                    for j in range(len(g_w)):
                        w_gradients[j] += g_w[j]
                        b_gradients[j] += g_b[j]

                for i, layer in enumerate(self.layers):
                    layer.w -= (w_gradients[i] / len(batch_x)) * learning_rate
                    layer.b -= (b_gradients[i] / len(batch_x)) * learning_rate

                if batch_start % (batch_size / 2) == 0: print(f"\r Cost : {cost / len(batch_x)}", end = '')

    def forward(
            self,
            X : np.ndarray
    ) -> list[np.ndarray]:
        """
        Passes through the network to calculate and store the activations of each layer.

        :param X: Input features, shape (n_samples, n_features)
        :return:
        """
        activations = [X]
        a_i = X
        for layer in self.layers:
            a_i = layer.compute(a_i)
            activations.append(a_i)

        return activations


    def backward(
            self,
            activations : list[np.ndarray],
            y : np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Calculates the gradient of every layer using back propagation.

        :param activations: List of activations for every layer
        :param y: Desired outputs, shape (n_samples, n_outputs)
        :return: List of gradients for both weights and biases
        """
        err = self.loss_deriv(activations[-1], y)

        w_gradients = []

        b_gradients = []


        for i in range(len(activations) - 1, 0, -1):
            activations_deriv = self.layers[i - 1].activation_deriv(activations[i])

            w_g = calc_w_gradients(activations[i - 1], activations_deriv, err)
            b_g = calc_b_gradients(activations_deriv,err)

            w_gradients.append(w_g)
            b_gradients.append(b_g)

            err = calc_back_signals(self.layers[i - 1].w, activations_deriv, err)

        return w_gradients[::-1], b_gradients[::-1]

def calc_back_signals(
        w: np.ndarray,
        activation_deriv: np.ndarray,
        error: np.ndarray
) -> np.ndarray:
    """
    Calculates the errors for layer l - 1.

    :param w: Weight matrix of the current layer
    :param activation_deriv: Derivatives of the activations of the current layer
    :param error: Error of the current layer
    :return: Vector of error for layer l - 1
    """

    return w @ (activation_deriv * error)

def calc_w_gradients(
        activations: np.ndarray,
        activation_deriv: np.ndarray,
        error: np.ndarray
) -> np.ndarray:
    """
    Calculates the weight gradients using the chain rule.

    :param activations: Activations for layer l - 1
    :param activation_deriv: Derivative of the activations of the current layer
    :param error: Error of the current layer
    :return: Vector of the gradients with respect to each weight in the layer
    """
    return activations.reshape(-1, 1) @ (activation_deriv * error).reshape(1, -1)

def calc_b_gradients(
        activation_deriv: np.ndarray,
        error: np.ndarray
) -> np.ndarray:
    """
    Calculates the bias gradients using the chain rule.

    :param activation_deriv: Derivative of the activations of the current layer
    :param error: Error of the current layer
    :return: Vector of the gradients with respect to each bias in the layer
    """
    return activation_deriv * error







