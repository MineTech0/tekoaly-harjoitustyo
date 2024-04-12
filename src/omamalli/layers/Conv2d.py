import numpy as np
from .BaseLayer import BaseLayer

class Conv2D(BaseLayer):
    def __init__(self, num_filters: int, kernel_size: int, input_shape: tuple, stride=1, learning_rate=0.01):
        # Initialize the parent class
        super().__init__()
        
        # Initialization of layer attributes
        self.num_filters = num_filters  # Number of filters in the convolutional layer
        self.kernel_size = kernel_size  # Size of each filter (assumed square)
        self.stride = stride  # Stride size used during the convolution
        self.learning_rate = learning_rate  # Learning rate for the optimization
        
        # Filters dimensions: (num_filters, kernel_size, kernel_size, input_channels)
        _, _, input_channels = input_shape
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size, input_channels) * np.sqrt(2. / (kernel_size * kernel_size * input_channels))
        
        # Initialize biases for each filter
        self.biases = np.zeros(num_filters)
        
        # Store input and output shapes
        self.input_shape = input_shape  # Expected shape of the input
        self.output_shape = self.compute_output_shape(input_shape)  # Calculated shape of the output
        
        # Placeholder for caching the input during the forward pass for use in the backward pass
        self.cache_input = None

    def compute_output_shape(self, input_shape):
        # Calculate the dimensions of the output volume
        height, width, _ = input_shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        return (output_height, output_width, self.num_filters)

    def forward(self, input_array, training=False):
        # Cache the input array for use in the backward pass
        self.cache_input = input_array
        batch_size, input_height, input_width, input_channels = input_array.shape
        output_height, output_width, _ = self.output_shape
        
        # Initialize the output volume with zeros
        output = np.zeros((batch_size, output_height, output_width, self.num_filters))
        
        # Convolve the filter over the input image
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(0, output_height):
                    for x in range(0, output_width):
                        # Extract the current region of interest
                        input_region = input_array[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, :]
                        # Perform the convolution operation and add the bias
                        output[i, y, x, f] = np.sum(input_region * self.filters[f]) + self.biases[f]
        return output

    def backward(self, d_output):
        # Prepare gradients for filters, biases, and input
        batch_size, input_height, input_width, input_channels = self.cache_input.shape
        _, output_height, output_width, _ = d_output.shape
        
        d_filters = np.zeros(self.filters.shape)
        d_biases = np.zeros(self.biases.shape)
        d_input = np.zeros(self.cache_input.shape)
        
        # Backpropagation through the convolutional layer
        for i in range(batch_size):
            for f in range(self.num_filters):
                for y in range(output_height):
                    for x in range(output_width):
                        input_region = self.cache_input[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, :]
                        # Gradient w.r.t. filter weights
                        d_filters[f] += d_output[i, y, x, f] * input_region
                        # Gradient w.r.t. biases
                        d_biases[f] += d_output[i, y, x, f]
                        # Gradient w.r.t. input of the layer
                        d_input[i, y*self.stride:y*self.stride+self.kernel_size, x*self.stride:x*self.stride+self.kernel_size, :] += d_output[i, y, x, f] * self.filters[f]
        
        # Update the filters and biases using the calculated gradients
        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases
        return d_input
