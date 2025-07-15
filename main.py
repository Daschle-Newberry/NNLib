import matplotlib.pyplot as plt
import numpy as np

from util.data import load_rand_circle, load_coffee
from neuralnet.layer import Layer
from neuralnet.neuralnetwork import NeuralNetwork
from util.plotutils import plot_2d_scatter, plot_2d_heatmap

# import tensorflow as tf
# from keras.layers import Dense
# from keras.models import Sequential

def main():
    X,y = load_coffee()

    X = (X - X.mean()) / X.std()


    network = NeuralNetwork(
            [
                Layer(3,'sigmoid'),
                Layer(1,"sigmoid")
            ],'BCE'
    )

    network.compile(input_size = 2)

    print(network)
    network.fit_stochastic(X,y,learning_rate = .01 , epochs = 10000)



    fig, ax = plt.subplots(1, 1, figsize=(10, 20))

    plot_2d_heatmap(X, network, ax)
    plot_2d_scatter(X,y,ax)

    plt.show()



if __name__ == "__main__":
    main()