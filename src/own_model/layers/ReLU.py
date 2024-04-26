import numpy as np
from .BaseLayer import BaseLayer

class ReLU(BaseLayer):
    def initialize(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = input_shape
    
    def forward(self, input_array: np.ndarray, training=False) -> np.ndarray:
        self.input = input_array
        self.output = np.maximum(0, input_array)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient
