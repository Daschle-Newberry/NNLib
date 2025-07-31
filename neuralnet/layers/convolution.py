import numpy as np

from neuralnet.layers.layer import TrainableLayer


class Convolution(TrainableLayer):
    def __init__(self, kernel_size: int, output_channels: int, mode: str):

        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.mode = mode

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

        self._kernels = np.random.rand(self.output_channels, input_channels, kernel_height, kernel_width)

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
        self._output = np.zeros(self.output_dim)
        for i in range(self.num_kernels):
            for j in range(self.num_input_channels):
                self._output[i] += correlate2d(x[j], self._kernels[i, j], self.mode)
            self._output[i] += self._biases[i]
        return self._output

    def backward(self, output_gradients: np.ndarray):
        input_gradients = np.zeros_like(self._input).astype(np.float64)

        for i in range(self.num_kernels):
            # J corresponds to the channel, while I corresponds to the kernel, so Kij and Xj, and Yi
            for j in range(self.num_input_channels):
                self._dK[i, j] += correlate2d(self._input[j], output_gradients[i], 'valid')
                input_gradients[j] += correlate2d(output_gradients[i], self._kernels[i, j], 'full')
            self._dB[i] += output_gradients[i].sum()
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

    @property
    def b_gradients(self):
        return self._dB

    @property
    def output_dim(self):
        return self._output_dim


def correlate2d(x_input: np.ndarray, kernel: np.ndarray, mode: str):
    kH, kW = kernel.shape

    x = None
    if mode == 'full':
        pH, pW = (kH - 1, kW - 1)
        x = np.pad(x_input, pad_width=((pH, pH), (pW, pW)), mode='constant', constant_values=0)
    elif mode == 'valid':
        x = x_input

    iH, iW = x.shape

    oH = iH - kH + 1
    oW = iW - kW + 1

    windows = np.lib.stride_tricks.sliding_window_view(x, (kH, kW))

    window_reshaped = windows.reshape(oH, oW, -1)
    kernel_reshaped = kernel.reshape(kH * kW)

    output = np.tensordot(window_reshaped, kernel_reshaped, axes=([2], [0]))

    return output
