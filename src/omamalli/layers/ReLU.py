import numpy as np
from .BaseLayer import BaseLayer

class ReLU(BaseLayer):
    def forward(self, input_array: np.ndarray) -> np.ndarray:
        self.input = input_array
        self.output = np.maximum(0, input_array)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient
