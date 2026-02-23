import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)

import json

import numpy as np
from NNLib.models import Sequential
from util.mnist_data import load_data
from NNLib.layers import Convolution, MaxPool, Flatten, Dense
from NNLib.layers.activations import ReLu

from NNLib.losses import SoftMaxCrossEntropy
from NNLib.optimizers import Adam
from ui import Window, DrawingBoard, Button, NetworkUIManager


def main():

    (X, y), (Xt, yt) = load_data()

    X = X.reshape(-1, 1, 28, 28)


    Xt = Xt.reshape(-1, 1, 28, 28)

    X_mean = X.mean()
    X_std = X.std()

    X = (X - X_mean) / X_std

    Xt = (Xt - X_mean) / X_std

    model = Sequential(
        [
            Convolution((3, 3), 8, 'valid', 'layer0'),
            ReLu(),
            MaxPool(2, 2),
            Convolution((3, 3), 16, 'valid', 'layer1'),
            ReLu(),
            MaxPool(2, 2),
            Flatten(),
            Dense(128, 'layer2'),
            ReLu(),
            Dense(10, 'layer3'),
        ], SoftMaxCrossEntropy()
    )

    model.compile(input_size=(1, 28, 28), optimizer=Adam(learning_rate=.001, beta1=.9, beta2=.999))



    print("Would you like to train a fresh model? (y/n)")
    response = input()

    if response == 'y':
        print("Training can take several minutes...")
        model.fit(X, y, epochs = 100, batch_size = 64)

    else:
        with open("examples/data/mnist_params.json", "r") as f:
            params_loaded = json.load(f)

        params_loaded = {
            k : [np.array(arr) for arr in v] for k, v in params_loaded.items()
        }

        model.set_params(params_loaded)

    print("Testing model against mnist test samples...")
    prediction = np.argmax(model.predict(Xt), axis=1)
    model_accuracy = np.mean(prediction == yt)

    print(f"Model accuracy is {model_accuracy}")

    print("You may now draw digits in the white box. When you are ready to submit, press the blue button.")
    print("Please be aware that the model has high bias towards the center, so numbers written off center or smaller may not be classified correctly")
    window = Window(725, 725)

    db = DrawingBoard((300, 300), (362, 300))
    window.add_component(db)

    btn = Button((50, 50), (362, 600), (0, 0, 255))
    window.add_component(btn)

    manager = NetworkUIManager(model, db, btn, X_mean, X_std)

    window.run()

if __name__ == "__main__":
    main()