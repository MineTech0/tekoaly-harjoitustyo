import numpy as np
from .BaseLayer import BaseLayer


class Dense(BaseLayer):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_shape = (output_size,)
        self.weights = None
        self.biases = None

    def initialize(self, input_shape):
        self.input_shape = input_shape
        input_size = input_shape[0]
        # Initialize weights with dimensions (number of input features, number of output features)
        self.weights = np.random.randn(input_size, self.output_shape[0]) * 0.01
        self.biases = np.zeros(self.output_shape)

    def forward(self, input_array: np.ndarray, training=False) -> np.ndarray:
        self.input = input_array
        self.output = np.dot(input_array, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # Compute and return input gradient
        return np.dot(output_gradient, self.weights.T)
