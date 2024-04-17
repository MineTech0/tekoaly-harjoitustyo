import numpy as np
from .BaseLayer import BaseLayer
import scipy

class Conv2D(BaseLayer):
    def __init__(self, num_filters: int, kernel_size: int, input_shape: tuple = None, stride=1):
        # Initialize the parent class
        super().__init__()
        
        # Initialization of layer attributes
        self.num_filters = num_filters  # Number of filters in the convolutional layer
        self.kernel_size = kernel_size  # Size of each filter (assumed square)
        self.stride = stride  # Stride size used during the convolution
        
        # Filters dimensions: (num_filters, kernel_size, kernel_size, input_channels)
        if input_shape is not None:
            _, _, input_channels = input_shape
            self.filters = np.random.randn(num_filters, kernel_size, kernel_size, input_channels) * np.sqrt(2. / (kernel_size * kernel_size * input_channels))
            
            # Store input and output shapes
            self.input_shape = input_shape  # Expected shape of the input
            self.output_shape = self.compute_output_shape(input_shape)  # Calculated shape of the output
        
        # Initialize biases for each filter
        self.biases = np.zeros(num_filters)
        
        # Placeholder for caching the input during the forward pass for use in the backward pass
        self.cache_input = None
        
    def initialize(self, input_shape=None):
        # Initialize the filters and biases
        if self.input_shape is None:
            _, _, input_channels = input_shape
            self.filters = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size, input_channels) * np.sqrt(2. / (self.kernel_size * self.kernel_size * input_channels))
            self.output_shape = self.compute_output_shape(input_shape)
            self.input_shape = input_shape
            
    def compute_output_shape(self, input_shape):
        # Calculate the dimensions of the output volume
        height, width, _ = input_shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        return (output_height, output_width, self.num_filters)

    def forward(self, input_array, training=False):
        #check if input shape is correct
        if input_array.shape[1:] != self.input_shape:
            raise ValueError(f"Input shape {input_array.shape[1:]} does not match expected shape {self.input_shape}")
        
        batch_size, _, _, channels = input_array.shape
        output_height, output_width, num_filters = self.output_shape
        
        # Initialize output array
        output = np.zeros((batch_size, output_height, output_width, num_filters))
        
        self.cache_input = input_array

        # Perform the convolution
        for i in range(batch_size):
            for f in range(num_filters):
                for c in range(channels):
                    temp_result = scipy.signal.convolve(input_array[i, :, :, c], self.filters[f, :, :, c], mode='valid')
                
                # Apply stride
                if self.stride > 1:
                    temp_result = temp_result[::self.stride, ::self.stride]
                
                output[i, :, :, f] = temp_result + self.biases[f]
                

        return output

    def backward(self, d_output, learning_rate):
        # Prepare gradients for filters, biases, and input
        batch_size = self.cache_input.shape[0]
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
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        return d_input
