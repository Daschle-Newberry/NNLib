import numpy as np
from NNLib.layers.base import TrainableLayer


class Convolution(TrainableLayer):
    def __init__(self, kernel_size: tuple, output_channels: int, mode: str, name : str):

        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.mode = mode
        self.name = name

        self._kernels = None

        self._biases = None

        self.num_kernels, self.num_input_channels, self.kernel_height, self.kernel_width = (None, None, None, None)

        self._input_dim = None

        self._output_dim = None

        self._dK, self._dB = None, None

        self._input = None
        self._output = None

    def compile(self, input_dim: tuple):
        input_channels, input_height, input_width = input_dim
        kernel_height, kernel_width = self.kernel_size

        fan_in = input_channels * kernel_height * kernel_width
        std = np.sqrt(2.0/fan_in)
        self._kernels = np.random.rand(self.output_channels, input_channels, kernel_height, kernel_width) * std

        self._biases = np.random.rand(self.output_channels)

        self.num_kernels, self.num_input_channels, self.kernel_height, self.kernel_width = self._kernels.shape

        self._input_dim = input_dim

        if self.mode == 'full':
            output_width = input_width + kernel_width - 1
            output_height = input_height + kernel_height - 1
        elif self.mode == 'valid':
            output_width = input_width - kernel_width + 1
            output_height = input_height - kernel_height + 1
        else:
            raise ValueError(f"Unknown padding mode {self.mode}")

        self._output_dim = (self.output_channels, output_height, output_width)

        self._dK = np.zeros_like(self._kernels)
        self._dB = np.zeros_like(self._biases)


    def forward(self, x: np.ndarray):
        self._input = x
        self._output = np.zeros((len(x), *self.output_dim))

        self._output += correlate(x, self._kernels, 'valid',False)


        return self._output + self._biases.reshape(1,-1,1,1)

    def backward(self, output_gradients: np.ndarray):

        input_gradients = np.zeros_like(self._input).astype(np.float64)

        self._dK += correlate(self._input, output_gradients,'valid', True)

        kernels_flipped = np.flip(self._kernels.transpose(1,0,2,3), axis = (2,3))
        input_gradients += correlate(output_gradients, kernels_flipped,'full',False)

        self._dB += output_gradients.sum(axis = (0,2,3))

        return input_gradients


    @property
    def weights(self):
        return self._kernels

    @weights.setter
    def weights(self, w: np.ndarray):
        self._kernels[:] = w

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, b: np.ndarray):
        self._biases[:] = b

    @property
    def w_gradients(self):
        return self._dK

    @w_gradients.setter
    def w_gradients(self, dW : np.ndarray):
        self._dK[:] = dW

    @property
    def b_gradients(self):
        return self._dB

    @b_gradients.setter
    def b_gradients(self, dB: np.ndarray):
        self._dB[:] = dB

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def dK(self):
        return self._dK

def correlate(x : np.ndarray, kernel : np.ndarray, mode : str, per_batch_kernel : bool):

    if not per_batch_kernel:
        kernel_output_channels, kernel_input_channels, kernel_height, kernel_width = kernel.shape
    else:
        kernel_batch_size, kernel_input_channels, kernel_height, kernel_width = kernel.shape
    if mode == 'full':
        pH, pW = (kernel_height - 1, kernel_width - 1)
        x = np.pad(x, pad_width=((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant', constant_values=0)
    elif mode == 'valid':
        pass

    windows = np.lib.stride_tricks.sliding_window_view(x, (1,1,kernel_height,kernel_width))

    windows = windows.squeeze(axis = (4,5))

    if per_batch_kernel:
        res = np.tensordot(windows,kernel, axes = ([0,4,5],[0,2,3])).transpose(3,0,1,2)
    else:
        res = np.tensordot(windows, kernel, axes=([1, 4, 5], [1, 2, 3])).transpose(0,3,1,2)
    return res

