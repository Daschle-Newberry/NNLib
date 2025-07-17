import matplotlib.pyplot as plt
import numpy as np

from neuralnet.optimizers import Optimizer, Momentum, Basic, RMSProp
from util.data import load_rand_circle, load_coffee, load_rand_circles
from neuralnet.layer import Layer
from neuralnet.neuralnetwork import NeuralNetwork
from util.plotutils import plot_2d_scatter, plot_2d_heatmap


def main():
    X,y = load_rand_circles(12,.05,.25)
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)

    X = (X - mean) / std
    X = np.tile(X,(1000,1))
    y = np.tile(y, (1000,))


    print(len(X))
    rms = NeuralNetwork(
        [
            Layer(16, 'sigmoid'),
            Layer(8, 'sigmoid'),
            Layer(1, "sigmoid")
        ], 'BCE'
    )

    rms.compile(input_size=2, optimizer=RMSProp(learning_rate=.01,rho = .99))

    rms.fit(X, y, epochs=10)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    plot_2d_heatmap(X, rms, ax[0])
    plot_2d_scatter(X, y, ax[0])

    plot_2d_scatter(X, y, ax[1])

    plt.show()



if __name__ == "__main__":
    main()