import numpy as np

from NNLib.layers import Layer, TrainableLayer
from NNLib.logging import TrainingMonitor
from NNLib.losses import LossFunction
from NNLib.optimizers import Optimizer

class Sequential:
    """
    A sequential network class which manages the layer interactions, training, and evaluation of a deep learning model.
    """

    def __init__(self, layers : list[Layer], loss : LossFunction):
        """
        Creates a network object which contains the methods and layers required to train and use a deep learning network.

        Parameters
        ----------
        layers : list[base]
            A python list containing layer objects which define the architecture of the network.

        loss : LossFunction
            The loss function used for the network.

        """
        self.layers = layers
        self.loss = loss
        self.optimizer = None
        self.input_size = None

    def compile(self, input_size : tuple, optimizer : Optimizer):
        """
        Compiles the network by linking each layers output to the next layers input.

        The compile function also sets up the optimizer by linking it with references of the network parameters.

        Parameters
        ----------
        input_size : tuple
            Tuple containing the expected input dimensions for the network

        optimizer : Optimizer
            An instance of an optimizer which will be used for network parameter updates
        """

        self.optimizer = optimizer
        self.input_size = input_size
        for i, layer in enumerate(self.layers):
            input_dim = input_size if i == 0 else self.layers[i - 1].output_dim
            layer.compile(input_dim)
        self.optimizer.compile(self.get_params())


    def predict(
                self,
                X : np.ndarray
    ) -> np.ndarray:
        """
        Runs a singular forward pass on the input data, data to be expected in batch format.

        This function calls _forward_pass() internally as they serve the same function. This function is intended for external use.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, ...)
          List of input data samples. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.


        Returns
        ----------
        np.ndarray, shape (n_samples,...)
            The model's predictions for each input sample.
        """
        return self._forward_pass(X)

    def fit(
            self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            batch_size : int = 32
    ) -> list[float]:
        """
        Trains model using batched gradient descent. Model weights and biases are updated after each batch.

        Parameters
        ----------

        X: np.ndarray, shape (n_samples, ....)
                List of training input data. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.

        y : np.ndarray, shape (n_samples, ....)
            Desired network outputs. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.

        epochs : int
            Amount of iterations over the training data.

        batch_size : int, optional
            Number of samples per gradient update. Automatically set to 32 if not specified.

        Returns
        ----------
        list[float]
            A history of the cost during training, as specified by the training logger used.
        """

        logger = TrainingMonitor(epochs,int(len(X) / batch_size))

        for epoch in range(1,epochs):

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]


            for batch_start in range(0, len(X), batch_size):
                batch_x = X[batch_start : batch_start + batch_size]
                batch_y = y[batch_start : batch_start + batch_size]

                a = self._forward_pass(batch_x)

                cost = self.loss.forward(a,batch_y)

                self._backward_pass(a, batch_y)

                self.optimizer.step(len(batch_x))

                is_final = batch_start >= len(X) - batch_size - 1 and epoch == epochs - 1
                logger.update(is_final, epoch = epoch,batch = np.ceil(batch_start / batch_size), cost = cost)

        return logger.get_cost_history()


    def _forward_pass(
            self,
            X : np.ndarray
    ) -> np.ndarray:
        """
        Runs a forward pass on the input data, data to be expected in batch format.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, ...)
          List of input data samples. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.

        Returns
        ----------
        np.ndarray, shape (n_samples,...)
            The model's predictions for each input sample.
        """
        if X[0].shape != self.input_size:
            raise ValueError(f"Input size {self.input_size} does not match actual input size {X[0].shape}")
        for layer in self.layers:
            X = layer.forward(X)

        return X


    def _backward_pass(
            self,
            yHat : np.ndarray,
            y : np.ndarray
    ) -> None:
        """
        Runs a backwards pass to calculate and store the gradients of each layer.

        Parameters
        ----------
        yHat : np.ndarray, shape (n_samples, ....)
            Predicted network outputs. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.

        y : np.ndarray, shape (n_samples, ....)
            Desired network outputs. Axis 0 should be the batch dimension, all other dimensions should align with the expected input dimensions.

        """


        output_gradient = self.loss.backward(yHat, y)

        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)


    def _debug_set_params_one(self):
        """
        Initializes all network parameters to one, useful for debugging
        """

        for layer in self.layers:
            if not isinstance(layer,TrainableLayer):
                continue
            layer.weights = np.ones_like(layer.weights)
            layer.biases = np.ones_like(layer.biases)

    def _debug_forward_pass_from(self, x : np.ndarray, start : int):
        """
        Starts a forward pass from a specified layer
        """
        layers = self.layers[start:]

        X = x
        for layer in layers:
            X = layer.forward(X)

        return X

    def _debug_backward_pass(self, yHat : np.ndarray, y : np.ndarray):
        """
        Calculates the gradients of the backward pass and stores them in list format, useful for debugging.
        """

        if yHat.shape != y.shape:
            raise ValueError(f"Shape mismatch between yHat and y, is your training data correct? {yHat.shape},{y.shape}")

        output_gradient = self.loss.backward(yHat, y)

        output_gradients = []
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
            output_gradients.append(output_gradient)

        return output_gradients[::-1]

    def _debug_analytical_gradients(self, x : np.ndarray, y : np.ndarray):
        """
        Calculates the analytical gradients for a single batch, useful for debugging.
        """
        for layer in self.layers:
            if not isinstance(layer,TrainableLayer):
                continue
            layer.w_gradients = np.zeros_like(layer.w_gradients)
            layer.b_gradients = np.zeros_like(layer.b_gradients)

        a = self.forward_pass(x)
        input_gradients = self.debug_backward_pass(a,y)

        return input_gradients

    def _debug_numerical_gradients(self, x : np.ndarray, y : np.ndarray):
        """
        Calculates the numerical gradients for a single batch, useful for debugging.
        """
        e = 1e-6

        input_gradients = []
        X = x
        for l, layer in enumerate(self.layers):
            X_gradients = np.zeros_like(X).reshape(-1)
            X_shape = X.shape
            X = X.reshape(-1)

            for i in range(len(X)):
                X[i] += e
                y_hat_1 = self.debug_forward_pass_from(X.reshape(X_shape),l)

                X[i] -= 2 * e
                y_hat_2 = self.debug_forward_pass_from(X.reshape(X_shape),l)

                loss_1 = self.loss.forward(y_hat_1, y)
                loss_2 = self.loss.forward(y_hat_2, y)

                X_gradients[i] = (loss_1 - loss_2) / (2 * e)

                X[i] += e

            X = layer.forward(X.reshape(X_shape))

            input_gradients.append(X_gradients.reshape(X_shape))


        for layer in self.layers:
            if not isinstance(layer,TrainableLayer):
                continue

            weights = layer.weights
            biases = layer.biases
            weight_gradients = layer.w_gradients
            bias_gradients = layer.b_gradients

            weights = weights.reshape(-1)
            biases = biases.reshape(-1)
            weight_gradients = weight_gradients.reshape(-1)
            bias_gradients = bias_gradients.reshape(-1)



            for i in range(len(weights)):
                weights[i] += e
                y_hat_1 = self.forward_pass(x)

                weights[i] -= 2 * e
                y_hat_2  = self.forward_pass(x)

                loss_1 = self.loss.forward(y_hat_1,y)
                loss_2 = self.loss.forward(y_hat_2,y)


                weight_gradients[i] = (loss_1 - loss_2) / (2 * e)

                weights[i] += e

            for i in range(len(biases)):
                biases[i] += e
                y_hat_1 = self.forward_pass(x)

                biases[i] -= 2 * e
                y_hat_2 = self.forward_pass(x)

                loss_1 = self.loss.forward(y_hat_1, y)
                loss_2 = self.loss.forward(y_hat_2, y)


                bias_gradients[i] = (loss_1 - loss_2) / (2 * e)

                biases[i] += e


        return input_gradients




    def _debug_get_params_copy(self):
        """
        Returns a copy of each parameter, useful for debugging
        """
        return {"w" : [layer.weights.copy() for layer in self.layers if isinstance(layer,TrainableLayer)],
                "b" : [layer.biases.copy() for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dW" : [layer.w_gradients.copy() for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dB" : [layer.b_gradients.copy() for layer in self.layers if isinstance(layer,TrainableLayer)]
                }


    def set_params(self, params : dict[str, list[np.ndarray]]):
        w = params.get('w')
        b = params.get('b')

        trainable_layers = [layer for layer in self.layers if isinstance(layer, TrainableLayer)]

        for layer,w,b in zip(trainable_layers,w,b):
            layer.weights = w
            layer.biases = b

    def get_params(self):
        """
        Generates a dictionary of references to each parameter, useful for debugging

        Returns
        ----------
        dict[str,np.ndarray]
            Dictionary with keys corresponding to each parameter type, and values as references to the parameters.
        """
        return {"w" : [layer.weights for layer in self.layers if isinstance(layer,TrainableLayer)],
                "b" : [layer.biases for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dW" : [layer.w_gradients for layer in self.layers if isinstance(layer,TrainableLayer)],
                "dB" : [layer.b_gradients for layer in self.layers if isinstance(layer,TrainableLayer)]
                }

    def get_layer(self, index : int):
        """
        Gets a layer at a specific index in the network

        index : int
            Index of the layer to return

        Returns
        ----------
        base
            The layer at the index
        """
        return self.layers[index]




