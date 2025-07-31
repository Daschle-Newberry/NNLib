import numpy as np

from keras.datasets import mnist

from ui.button import Button
from ui.drawingboard import DrawingBoard
from ui.uimanager import NetworkUIManager
from ui.window import Window
from neuralnet.layers.convolution import Convolution
from neuralnet.layers.flatten import Flatten
from neuralnet.losses.losses import BinaryCrossEntropy
from neuralnet.optimizers import RMSProp, SGD, Momentum
from neuralnet.layers.dense import Dense
from neuralnet.neuralnetwork import NeuralNetwork
from neuralnet.layers.activations.relu import ReLu
from neuralnet.layers.activations.sigmoid import Sigmoid



def main():
    np.seterr(over='raise', divide='raise', invalid='raise')
    np.random.seed(1234)

    (X, y), (Xt, yt) = mnist.load_data()

    filter = (y == 0) | (y == 1)

    X = X[filter]
    y = y[filter]

    y = y.astype(np.float32)

    X_mean = X.mean()
    X_std = X.std()
    X = (X - X.mean()) / X.std()

    X = X.reshape((-1, 1, 28, 28))
    y = y.reshape(-1,1)

    rms = NeuralNetwork(
        [
            Convolution((3,3),2,'valid'),
            ReLu(),
            Flatten(),
            Dense(64,'layer1'),
            ReLu(),
            Dense(1, 'layer1'),
            Sigmoid()
        ], BinaryCrossEntropy()
    )


    rms.compile(input_size=(1,28,28), optimizer = Momentum(learning_rate=.01, beta = .99))


    rms.fit(X,y, epochs = 10)


    print(X[0].shape)



    window = Window(725, 725)

    db = DrawingBoard((500, 500), (362, 300))
    window.add_component(db)

    btn = Button((50, 50), (362, 600), (0, 0, 255))
    window.add_component(btn)

    manager = NetworkUIManager(rms,db,btn, X_mean, X_std)

    window.run()




if __name__ == "__main__":
    main()