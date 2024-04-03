import numpy as np
from .BaseLayer import BaseLayer

class Conv2D(BaseLayer):
    def __init__(self, num_filters, kernel_size, input_shape, stride=1, learning_rate=0.01):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.learning_rate = learning_rate
        
        # He initialization for the weights
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) * np.sqrt(2. / (kernel_size * kernel_size))
        
        self.input_shape = input_shape  # Assuming (height, width, channels=1)
        self.output_shape = self.compute_output_shape(input_shape)
        
        # Cache for backward pass
        self.cache_input = None

    def compute_output_shape(self, input_shape):
        height, width, _ = input_shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        return (output_height, output_width, self.num_filters)

    def forward(self, input_array):
        self.cache_input = input_array
        batch_size, input_height, input_width, _ = input_array.shape
        output_height, output_width, _ = self.output_shape
        
        output = np.zeros((batch_size, output_height, output_width, self.num_filters))
        
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(output_height):
                    for x in range(output_width):
                        input_region = input_array[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, 0]
                        output[i, y, x, f] = np.sum(input_region * self.filters[f])
        return output

    def backward(self, d_output):
        batch_size, input_height, input_width, _ = self.cache_input.shape
        _, output_height, output_width, _ = d_output.shape
        
        d_filters = np.zeros(self.filters.shape)
        d_input = np.zeros(self.cache_input.shape)
        
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(output_height):
                    for x in range(output_width):
                        input_region = self.cache_input[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, 0]
                        d_filters[f] += d_output[i, y, x, f] * input_region
                        d_input[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, 0] += d_output[i, y, x, f] * self.filters[f]
        
        # Update filters
        self.filters -= self.learning_rate * d_filters
        return d_input
