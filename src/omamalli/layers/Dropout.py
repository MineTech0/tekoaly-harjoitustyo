import numpy as np
from .BaseLayer import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.output_shape = None
        
    def initialize(self, input_shape=None):
        self.output_shape = input_shape

    def forward(self, input_array, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_array.shape) / (1 - self.rate)
            return input_array * self.mask
        else:
            return input_array

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask
