import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from neuralnet.optimizers import Optimizer, Momentum, Basic, RMSProp
from util.data import load_rand_circle, load_coffee, load_rand_circles
from neuralnet.layer import Layer
from neuralnet.neuralnetwork import NeuralNetwork
from util.plotutils import plot_2d_scatter, plot_2d_heatmap


def main():
    np.seterr(over='raise', divide='raise', invalid='raise')

    (X,y), (Xt, yt) = mnist.load_data()

    filter = (y == 0) | (y == 1)

    X = X[filter]
    y = y[filter]

    y = y.astype(np.float32)
    X = (X - X.mean()) / X.std()

    filter_test = (yt == 0) | (yt == 1)
    Xt = Xt[filter_test]
    yt = yt[filter_test]

    print(yt.max())
    Xt = (Xt - Xt.mean())/ Xt.std()
    Xt = Xt.reshape((-1,28*28))

    X = X.reshape((-1,28*28))

    print(X.shape)

    rms = NeuralNetwork(
        [
            Layer(128, 'relu', "layer1"),
            Layer(64,'relu','layer2'),
            Layer(1, "sigmoid",'layer3')
        ], 'BCE'
    )

    rms.compile(input_size=28 * 28, optimizer=RMSProp(learning_rate=.01,rho = .99))

    rms.fit(X, y, epochs=10)


    for i in range(0,10):
        print(f"Test sample {i}, expected output {yt[i]}, got {rms.predict(Xt[i])}")




    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #
    # plot_2d_heatmap(X, rms, ax[0])
    # plot_2d_scatter(X, y, ax[0])
    #
    # plot_2d_scatter(X, y, ax[1])

    # plt.show()



if __name__ == "__main__":
    main()