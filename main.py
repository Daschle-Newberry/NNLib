import numpy as np
from layer import Layer
from neuralnetwork import NeuralNetwork

def main():


    arr = [1,2,3,4]

    for i in range(len(arr) - 2, -1 , -1):
        print(arr[i])


    network = NeuralNetwork(
            [
                Layer(2, 'sigmoid'),
                Layer(1, 'sigmoid')
            ]

    )

    network.compile(2)


    print(network.get_weights())
    pre, post = network.forward(np.array([10,10]))


    print("pre", pre, "\n", "post", post)

    network.backward(pre,post,np.array([1,1]))
    # print(network)





if __name__ == "__main__":
    main()