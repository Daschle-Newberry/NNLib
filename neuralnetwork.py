
import numpy as np
import numpy.random

import activations
import losses
from layer import Layer
class NeuralNetwork:
    def __init__(self, layers : list[Layer], loss_func : str):
        self.layers = layers
        self.loss = losses.get(loss_func)
        self.loss_deriv = losses.get_deriv(loss_func)
        numpy.random.seed(1234)

    def __str__(self):
        res = "LAYER ============= NEURONS ============= ACTIVATION\n"
        for i, layer in enumerate(self.network):
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


    def fit(self, X : np.ndarray,y : np.ndarray, learning_rate : float, epochs : int):
        for epoch in range(epochs):
            w_gradients = []
            b_gradients = []
            cost = 0
            for layer in self.layers:
                w_gradients.append(np.zeros_like(layer.w))
                b_gradients.append(np.zeros_like(layer.b))


            for i in range(len(X)):
                a = self.forward(X[i])

                cost += self.loss(a[-1], y[i])

                g_w, g_b = self.backward(a, y[i])

                for j in range(len(w_gradients)):
                    w_gradients[j] += (g_w[j] / len(X)) * learning_rate
                for j in range(len(b_gradients)):
                    b_gradients[j] += (g_b[j] / len(X)) * learning_rate



            for i, layer in enumerate(self.layers):
                layer.w -= w_gradients[i]
                layer.b -= b_gradients[i]

            print(f"Epoch : {epoch}, Cost : {cost / len(X)}")

        print("Weights ", self.get_weights())
        print("Bias ", self.get_biases())




    def forward(self, x : np.ndarray):
        activations = [x]
        a_i = x
        for layer in self.layers:
            a_i = layer.compute(a_i)
            activations.append(a_i)

        return activations


    def backward(self, activations : list[np.ndarray], y : np.ndarray):
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

def calc_back_signals(w: np.ndarray, activation_deriv: np.ndarray, error: np.ndarray):
    # print("----------------")
    # print(f"Activation Deriv {activation_deriv},{activation_deriv}")
    # print(f"Error {error},{error.shape}")
    #
    #
    # print(f"Activation Deriv * Error {activation_deriv * error},{(activation_deriv * error).shape}")
    # print(f"Weight {w.T},{w.T.shape}\n")

    return w @ (activation_deriv * error)

def calc_w_gradients(activations: np.ndarray, activation_deriv: np.ndarray, error: np.ndarray):
    # print(f"activations : {activations} {activations.shape}")
    # print(f"activations deriv: {activation_deriv}")
    # print(f"error : {error}\n")

    return activations.reshape(-1, 1) @ (activation_deriv * error).reshape(1, -1)

def calc_b_gradients(activation_deriv: np.ndarray, error: np.ndarray):
    # print(f"activations deriv: {activation_deriv}")
    # print(f"error : {error}\n")
    return activation_deriv * error
def calc_loss_deriv(yHat, y):
    return 2 * (yHat - y)

def SE(yHat : np.ndarray, y : np.ndarray):
    return (yHat - y) ** 2





