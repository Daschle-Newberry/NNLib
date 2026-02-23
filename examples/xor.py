import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)

import numpy as np
from NNLib.models import Sequential
from NNLib.layers import Dense
from NNLib.layers.activations import Sigmoid
from NNLib.losses import BinaryCrossEntropy
from NNLib.optimizers import Adam

def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        0,
        1,
        1,
        0
    ])

    y = y.reshape(-1, 1)

    X = np.tile(X, (10000, 1))
    y = np.tile(y, (10000, 1))

    model = Sequential(
        [
            Dense(3, "Dense1"),
            Sigmoid(),
            Dense(1, "Dense2"),
            Sigmoid(),
        ], BinaryCrossEntropy()
    )

    model.compile(input_size=(2,), optimizer=Adam(learning_rate=.01, beta1=.9, beta2=.99))

    model.fit(X, y, epochs=5, batch_size=32)

    prediction = (model.predict(X) > .5)
    model_accuracy = np.mean(prediction == y)

    print(f"Model accuracy is {model_accuracy}")


if __name__ == "__main__":
    main()