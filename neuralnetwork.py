from typing import Callable

import numpy as np
import numpy.random
from optree.functools import partial

from layer import Layer
class NeuralNetwork:

    def __init__(self, layers):
        self.network = layers
        numpy.random.seed(1234)

    def __str__(self):
        res = "LAYER ============= NEURONS ============= ACTIVATION\n"
        for i, layer in enumerate(self.network):
            res += f"{i}" + (6 - len(str(i))) * " " + "             " + f"{layer.units}" + (9 - len(str(layer.units))) * " " + "             " + f"{layer.activation_type}" + (11 - len(str(layer.activation_type))) * " " + "\n"

        return res
    def get_weights(self):
        weights = []
        for layer in self.network:
            weights.append(layer.w)
        return weights

    def get_biases(self):
        biases = []
        for layer in self.network:
            biases.append(layer.b)
        return biases


    def compile(self, input_size : int):
        for i, layer in enumerate(self.network):
            input_dim = input_size if i == 0 else self.network[i - 1].units
            layer.w = np.random.rand(input_dim, layer.units)
            layer.b = np.zeros(layer.units)

    def predict(self, x : np.ndarray):
        a_i = x
        for layer in self.network:
            a_i = layer.compute(a_i)
        return a_i

    def forward(self, x : np.ndarray):
        pre_activations = []
        post_activations = []
        a_i = x
        for layer in self.network:
            z_i = layer.pre_activation(a_i)
            a_i = layer.post_activation(z_i)

            pre_activations.append(z_i)
            post_activations.append(a_i)
        return pre_activations,post_activations


    def backward(self, pre_activations : np.ndarray, post_activations : np.ndarray, y : np.ndarray):
        err_sig_n = calc_output_gradient(post_activations[-1], y, self.network[-1].activation_deriv)

        error_signals = np.array((len(self.network)))

        error_signals[-1] = err_sig_n

        for i in range(len(self.network) - 2, -1, -1):
            layer = self.network[i]
            layer_next = self.network[i + 1]

            #Matrix of size nx1 (num of units) showing the gradient of a over the neurons output
            f_z = layer.activation_deriv(post_activations[i])

            #
            err_sig_next = error_signals[i + 1]


        return err_sig_n

def calc_output_gradient(
         post_activations : np.ndarray,
         y : np.ndarray,
         activation_deriv : Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:

    e_a = post_activations - y
    a_z = activation_deriv(post_activations)

    return e_a * a_z





