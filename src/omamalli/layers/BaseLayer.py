import numpy as np

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_array: np.ndarray) -> np.ndarray:
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
