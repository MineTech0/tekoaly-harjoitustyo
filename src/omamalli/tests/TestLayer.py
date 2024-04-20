import numpy as np

class TestLayer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = 100

    def initialize(self, input_shape=None):
        if input_shape:
            self.input_shape = input_shape

    def forward(self, input, training=False):
        return np.zeros(self.output_shape)

    def backward(self, grad, learning_rate):
        return grad  # simplistic gradient pass-through for testing
