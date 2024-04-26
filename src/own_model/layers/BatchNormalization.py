import numpy as np
from .BaseLayer import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.input_normalized = None
        

    def initialize(self, input_shape):
        # input_shape should be (width, height, filters) for a conv layer
        filters = input_shape[2]
        self.gamma = np.ones((1, 1, 1, filters))  # (1, 1, 1, filters) for broadcasting over batches, width, height
        self.beta = np.zeros((1, 1, 1, filters))
        self.running_mean = np.zeros((1, 1, 1, filters))
        self.running_var = np.ones((1, 1, 1, filters))
        self.output_shape = input_shape

    def forward(self, input_array, training=True):
        if training:
            # Compute the mean and variance across the batch and spatial dimensions
            batch_mean = np.mean(input_array, axis=(0, 1, 2), keepdims=True)
            batch_var = np.var(input_array, axis=(0, 1, 2), keepdims=True)

            # Update the running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.input_normalized = (input_array - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            self.input_normalized = (input_array - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * self.input_normalized + self.beta
        return self.output

    def backward(self, output_gradient, learning_rate):
        N, W, H, F = output_gradient.shape

        input_grad = (1 / (N * W * H)) * self.gamma / np.sqrt(self.running_var + self.epsilon) * (
            N * W * H * output_gradient - 
            np.sum(output_gradient, axis=(0, 1, 2), keepdims=True) -
            self.input_normalized * np.sum(output_gradient * self.input_normalized, axis=(0, 1, 2), keepdims=True)
        )

        d_gamma = np.sum(output_gradient * self.input_normalized, axis=(0, 1, 2), keepdims=True).squeeze()
        d_beta = np.sum(output_gradient, axis=(0, 1, 2), keepdims=True).squeeze()

        self.gamma -= learning_rate * d_gamma.reshape(self.gamma.shape)
        self.beta -= learning_rate * d_beta.reshape(self.beta.shape)

        return input_grad
