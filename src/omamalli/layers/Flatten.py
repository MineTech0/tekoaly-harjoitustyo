from .BaseLayer import BaseLayer

class Flatten(BaseLayer):
    def forward(self, input_array):
        self.input_shape = input_array.shape  # Cache to use in backward pass
        return input_array.reshape(input_array.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
