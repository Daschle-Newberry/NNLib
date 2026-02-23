import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)

import json

import matplotlib.pyplot as plt
import numpy as np

from NNLib.models import Sequential
from NNLib.layers import Dense
from NNLib.layers.activations import ReLu, Sigmoid
from NNLib.losses import BinaryCrossEntropy
from NNLib.optimizers import Adam
from util import plot_2d_scatter, plot_2d_heatmap


def main():
    np.seterr(over='raise', divide='raise', invalid='raise')
    np.random.seed(1234)

    data = np.loadtxt("examples/data/coffee_roasting.csv",delimiter = ",",skiprows = 1)

    X = data[:,:2]
    y = data[:,2:]

    X_tiled = np.tile(X,(10000,1))
    y_tiled = np.tile(y, (10000,1))

    X = (X - X.mean()) / X.std()
    X_tiled = (X_tiled - X_tiled.mean()) / X_tiled.std()

    model = Sequential([
        Dense(16,'layer0'),
        ReLu(),
        Dense(16, 'layer0'),
        ReLu(),
        Dense(1,'layer1'),
        Sigmoid()
    ],BinaryCrossEntropy()
    )

    model.compile((2,), optimizer = Adam(learning_rate = .001, beta1 = .9, beta2 = .99))


    print("Would you like to train a fresh model? (y/n)")
    response = input()

    if response == 'y':
        print("Training can take several minutes...")
        model.fit(X_tiled, y_tiled, epochs=30, batch_size=32)
    else:
        with open("examples/data/coffee_roasting_params.json", "r") as f:
            params_loaded = json.load(f)

        params_loaded = {
            k: [np.array(arr) for arr in v] for k, v in params_loaded.items()
        }

        model.set_params(params_loaded)

    print("Testing model against coffee samples...")
    prediction = model.predict(X) > .5
    model_accuracy = np.mean(prediction == y)

    print(f"Model accuracy : {model_accuracy}")

    fig, ax = plt.subplots(1,1,figsize = (8,8))

    plot_2d_heatmap(X,model,ax)
    plot_2d_scatter(X,y,ax)
    plt.show()


if __name__ == "__main__":
    main()