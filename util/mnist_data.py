import numpy as np
import urllib.request

def load_data():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    urllib.request.urlretrieve(url, 'mnist.npz')

    with np.load("mnist.npz") as data:
        X_train, y_train = data["x_train"], data["y_train"]
        X_test, y_test = data["x_test"], data["y_test"]


        return (X_train, y_train), (X_test, y_test)


