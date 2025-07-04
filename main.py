import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from layer import Layer
from neuralnetwork import NeuralNetwork

def main():


    x_train = np.array([[1000],[900],[800],[700],[600],[500], [400], [300], [200], [100]])

    y_train = np.array([0,0,0,0,0,0,1,1,1,1])

    model = Sequential([
            keras.Input(shape = (1,)),
            Dense(units = 1, activation = 'sigmoid'),
        ]
    )

    model.compile(loss = keras.losses.BinaryCrossentropy(), optimizer = keras.optimizers.Adam(learning_rate = .005))
    model.summary()

    model.fit(x_train, y_train, epochs = 100)

    network = NeuralNetwork(
        (
            Layer(1,  "sigmoid"),
        )
    )

    network.compile(3)

    w_1 = model.get_layer('dense').get_weights()[0]

    b_1 = model.get_layer('dense').get_weights()[1]


    network.network[0].w = w_1
    network.network[0].b = b_1


    print(network)


    print(f"TF got {model.predict(np.array([125]))}\n My network got {network.predict(np.array([125]))}")





if __name__ == "__main__":
    main()