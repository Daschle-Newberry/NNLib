import numpy as np

from keras.datasets import mnist
from matplotlib import pyplot as plt

from NNLib.layers import Dense, Convolution, MaxPool, Flatten, ReLu
from NNLib.losses import SoftMaxCrossEntropy
from NNLib.optimizers.adam import Adam

from NNLib.models import Sequential


def main():
    np.seterr(over='raise', divide='raise', invalid='raise')
    np.random.seed(1234)

    (X, y), (Xt, yt) = mnist.load_data()

    X = X.reshape(-1,1,28,28)

    Xt = Xt.reshape(-1,1,28,28)


    X = (X - X.mean()) / X.std()

    Xt = (Xt - X.mean()) / X.std()


    model = Sequential(
        [
            Convolution((3,3),8,'valid', 'layer0'),
            ReLu(),
            MaxPool(2,2),
            Convolution((3,3),16,'valid', 'layer1'),
            ReLu(),
            MaxPool(2,2),
            Flatten(),
            Dense(128,'layer3'),
            ReLu(),
            Dense(10, 'layer4'),
        ], SoftMaxCrossEntropy()
    )

    model.compile(input_size=(1,28,28), optimizer = Adam(learning_rate=.0001, beta1 = .9, beta2 = .99))

    cost_history = model.fit(X, y, epochs = 200, batch_size = 32)


    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost per epoch")
    plt.plot(cost_history)

    model_accuracy = test_model(model, Xt,yt)

    print(f"Model accuracy is {model_accuracy}")


    plt.show()
def test_model(model: Sequential, x_test, y_test):
    prediction = np.argmax(model.predict(x_test), axis = 1)
    accuracy = np.mean(prediction == y_test)

    return accuracy



if __name__ == "__main__":
    main()
