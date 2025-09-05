import time

import numpy as np

def main():
    X = np.random.randn(1, 3, 1000, 1000)

    K = np.random.randn(3, 3, 3, 3)

    start = time.time()
    y_vec = correlate(X, K, 'valid',False)
    end = time.time()

    print(f"Vectorized time {end - start}")

    start = time.time()
    convolution_naive(X,K)
    end = time.time()
    print(f"Naive time {end - start}")


def convolution_naive(x, kernel):
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape

    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1

    output =  np.zeros((batch,out_channels,output_height,output_width))

    for b in range(batch):
        print(f"Batch {b}")
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        output[b, oc, i, j] += np.sum(
                            x[b, ic, i:i+kernel_height, j:j + kernel_width] * kernel[oc, ic]
                        )


def correlate(x: np.ndarray, kernel: np.ndarray, mode: str, per_batch_kernel: bool):

    if not per_batch_kernel:
        kernel_output_channels, kernel_input_channels, kernel_height, kernel_width = kernel.shape
    else:
        kernel_batch_size, kernel_input_channels, kernel_height, kernel_width = kernel.shape
    if mode == 'full':
        pH, pW = (kernel_height - 1, kernel_width - 1)
        x = np.pad(x, pad_width=((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant', constant_values=0)
    elif mode == 'valid':
        pass

    windows = np.lib.stride_tricks.sliding_window_view(x, (1, 1, kernel_height, kernel_width))

    windows = windows.squeeze(axis=(4, 5))

    if per_batch_kernel:
        res = np.tensordot(windows, kernel, axes=([0, 4, 5], [0, 2, 3])).transpose(3, 0, 1, 2)
    else:
        res = np.tensordot(windows, kernel, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
    return res
main()