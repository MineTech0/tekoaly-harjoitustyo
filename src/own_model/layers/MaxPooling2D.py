import numpy as np
from numpy.lib.stride_tricks import as_strided
from .BaseLayer import BaseLayer

class MaxPooling2D(BaseLayer):
    def __init__(self, pool_size: int, stride: int = 1):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        self.output_shape = None

    def initialize(self, input_shape=None):
        input_height, input_width, input_channels = input_shape
        output_height = 1 + (input_height - self.pool_size) // self.stride
        output_width = 1 + (input_width - self.pool_size) // self.stride
        self.output_shape = (output_height, output_width, input_channels)

    def forward(self, input_array, training=False):
        self.input = input_array
        batch_size, input_height, input_width, input_channels = input_array.shape
        output_height, output_width, _ = self.output_shape
        output = np.zeros((batch_size, output_height, output_width, input_channels))

        # Store max indices for each pooling window
        self.max_indices = np.zeros((batch_size, output_height, output_width, input_channels), dtype=int)

        for n in range(batch_size):
            for c in range(input_channels):
                input_view = as_strided(
                    input_array[n, :, :, c],
                    shape=(output_height, output_width, self.pool_size, self.pool_size),
                    strides=(input_array.strides[1] * self.stride, input_array.strides[2] * self.stride, input_array.strides[1], input_array.strides[2])
                )
                max_idx = np.argmax(input_view.reshape(output_height, output_width, -1), axis=2)
                self.max_indices[n, :, :, c] = max_idx
                max_values = np.take_along_axis(input_view.reshape(output_height, output_width, -1), max_idx[:, :, None], axis=2).squeeze()
                output[n, :, :, c] = max_values

        self.cache = (input_array, output)
        return output

    def backward(self, output_gradient, learning_rate):
        input_array, _ = self.cache
        batch_size, input_height, input_width, input_channels = input_array.shape
        input_gradient = np.zeros_like(input_array)

        for n in range(batch_size):
            for c in range(input_channels):
                output_grad = output_gradient[n, :, :, c]
                indices_h = np.arange(output_grad.shape[0])[:, np.newaxis] * self.stride + self.max_indices[n, :, :, c] // self.pool_size
                indices_w = np.arange(output_grad.shape[1])[np.newaxis, :] * self.stride + self.max_indices[n, :, :, c] % self.pool_size
                input_gradient[n, indices_h, indices_w, c] += output_grad

        return input_gradient
