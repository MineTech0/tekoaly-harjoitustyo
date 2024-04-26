import numpy as np

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        
    def initialize(self, input_shape=None):
        """
        Initializes the layer with the input shape.
        Should be implemented by all subclasses.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, input_array: np.ndarray, training: bool) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        Should be implemented by all subclasses.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, output_gradient, learning_rate):
        """
        Computes the backward pass of the layer.
        Should be implemented by all subclasses.
        """
        raise NotImplementedError("Must be implemented by subclass.")
