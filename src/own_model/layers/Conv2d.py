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

    def backward(self, output_gradient, learning_rate):
        batch_size, _, _, channels = self.cache_input.shape
        
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(output_gradient, axis=(0, 1, 2))  # Sum over all except the filter axis
        d_input = np.zeros_like(self.cache_input)

        # Using scipy.signal.correlate to compute the gradient w.r.t inputs and update gradients w.r.t filters
        for i in range(batch_size):
            for f in range(self.num_filters):
                for c in range(channels):
                    # Rotate filter by 180 degrees to convert convolution to correlation
                    rotated_filter = np.rot90(self.filters[f, :, :, c], 2)
                    # Computing the gradient w.r.t input
                    d_input[i, :, :, c] += scipy.signal.correlate(output_gradient[i, :, :, f], rotated_filter, mode='full')
                    # Updating gradient w.r.t filters
                    d_filters[f, :, :, c] += scipy.signal.correlate(self.cache_input[i, :, :, c], output_gradient[i, :, :, f], mode='valid')

        # Updating the weights and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        
        return d_input
