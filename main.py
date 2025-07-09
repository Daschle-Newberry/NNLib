import numpy as np
import matplotlib.pyplot as plt

import activations
import data
# import tensorflow as tf

# from keras.models import Sequential
# from keras.layers import Dense
import keras

from layer import Layer
from neuralnetwork import NeuralNetwork
from plotutils import plot_2d_scatter


def main():
    print(activations.get('MSE'))

    X,y = data.load_coffee()

    X2 = X.flatten()

    n = (X2 - X2.min()) / (X2.max() - X2.min())

    X = n.reshape(X.shape)

    print(X)

    print(y)

    network = NeuralNetwork(
            [
                Layer(3,'sigmoid'),
                Layer(1,"sigmoid")
            ],'BCE'
    )

    network.compile(2)


    print(network.get_weights(),"\n")

    print(network.get_biases())

    network.fit(X,y,learning_rate = .1, epochs = 10000)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    plot_2d_scatter(X, y, ax[0])


    plt.show()


def calc_numerical_gradient(network : NeuralNetwork, x : np.ndarray, y : np.ndarray):
    epsilon = .000000000001
    yhat1 = network.predict(x)

    original_weight = network.layers[0].w[0][0]
    network.layers[0].w[0][0] = original_weight + epsilon

    l1 = SE(yhat1, y)

    network.layers[0].w[0][0] = original_weight - epsilon

    yhat2 = network.predict(x)

    l2 = SE(yhat2, y)



    network.layers[0].w[0][0] = original_weight

    return (l1 - l2) / (2 * epsilon)

def SE(yHat : np.ndarray, y : np.ndarray):
    return (yHat - y) ** 2
if __name__ == "__main__":
    main()