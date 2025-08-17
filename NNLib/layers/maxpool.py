import numpy as np

from NNLib.layers.base import UntrainableLayer


class MaxPool(UntrainableLayer):
    def __init__(self,window_size : int, stride : int):
        self._window_size = window_size
        self._stride = stride

        self._input_dim = None
        self._output_dim = None

        self._input = None
        self._output = None

        self._max_indices = None


    def compile(self, input_dim: tuple):
        self._input_dim = input_dim
        input_depth, input_height, input_width = input_dim

        output_height = ((input_height - self._window_size) // self._stride) + 1
        output_width = ((input_width - self._window_size) // self._stride) + 1

        self._output_dim = (input_depth, output_height, output_width)

    def forward(self, x: np.ndarray):
        self._input = x
        self._output = np.zeros((len(x), *self._output_dim))

        output,max_indices = pool(x,self._window_size,self._stride)

        self._output += output
        self._max_indices = max_indices

        return self._output

    @property
    def output_dim(self):
        return self._output_dim

    def backward(self, output_gradients: np.ndarray):
        batch_size, input_depth, input_height, input_width = self._input.shape
        batch_size, output_depth, output_height, output_width = self._output.shape

        input_gradients = np.zeros_like(self._input).reshape(-1)

        window_row_offsets, window_column_offsets = np.indices((output_height, output_width)) * self._stride

        row_idx = self._max_indices // self._window_size + window_row_offsets
        col_idx = self._max_indices % self._window_size + window_column_offsets

        flat_idx = (row_idx * input_width + col_idx).reshape(batch_size, -1)

        depth_offsets = np.arange(input_depth) * input_height * input_width
        depth_offsets = np.repeat(depth_offsets, output_height * output_width)

        flat_idx += depth_offsets
        flat_idx = flat_idx.reshape(-1)

        batch_offsets = np.arange(batch_size) * input_depth * input_height * input_width
        batch_offsets = np.repeat(batch_offsets, input_depth * output_height * output_width)

        flat_idx += batch_offsets
        output_gradients = output_gradients.reshape(-1)

        np.add.at(input_gradients, flat_idx, output_gradients)

        input_gradients = input_gradients.reshape(batch_size, input_depth, input_height, input_width)

        return input_gradients



def pool(x : np.ndarray, window_size : int, stride : int):
    batch_size, input_depth, input_height, input_width = x.shape

    output_height = ((input_height - window_size) // stride) + 1
    output_width = ((input_width - window_size) // stride) + 1

    windows = np.lib.stride_tricks.sliding_window_view(x,(window_size,window_size), axis = (2,3))

    windows_strided = windows[:,:,::stride,::stride,:,:]

    windows_reshaped = windows_strided.reshape(batch_size, input_depth,output_height, output_width, -1)

    output = np.max(windows_reshaped, axis = -1)
    max_indices = np.argmax(windows_reshaped, axis = -1)
    return output, max_indices