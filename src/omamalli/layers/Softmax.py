from .BaseLayer import BaseLayer
import numpy as np

class Softmax(BaseLayer):
    
    def forward(self, input_array, training=False):
        exps = np.exp(input_array - np.max(input_array, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient