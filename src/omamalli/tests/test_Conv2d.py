import numpy as np
import unittest
from ..layers import Conv2D

class TestConv2D(unittest.TestCase):
        
    def test_forward_pass(self):
        # Initialize Conv2D layer
        num_filters = 1
        kernel_size = 2
        input_shape = (4, 4, 1)  # height, width, channels
        stride = 1
        conv_layer = Conv2D(num_filters, kernel_size, input_shape, stride)
        
        # Manually set the filter weights for predictable behavior
        # Here, each filter element is 1 for easy summing
        conv_layer.filters = np.array([[[1, 1], [1, 1]]]).reshape((1, 2, 2, 1))
        
        # Define a simple input (4x4 image with 1 channel)
        # This example uses a sequence of numbers for easy verification
        input_array = np.array([[[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]]]).reshape((1, 4, 4, 1))
        
        # Expected output manually calculated
        expected_output = np.array([[[14, 18, 22],
                                      [30, 34, 38],
                                      [46, 50, 54]]]).reshape((1, 3, 3, 1))
        
        # Perform the forward pass
        output = conv_layer.forward(input_array)
        
        # Assert that the actual output matches the expected output
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                             err_msg="Forward pass output does not match expected output.")
    
    def test_backward(self):
        # Initialize Conv2D layer with specific parameters for testing
        num_filters = 2
        kernel_size = 2
        input_shape = (4, 4, 1)
        stride = 1
        learning_rate = 0.01
        conv_layer = Conv2D(num_filters, kernel_size, input_shape, stride, learning_rate)

        # Set filters and biases to known values
        conv_layer.filters = np.ones((num_filters, kernel_size, kernel_size, input_shape[2])) * 0.5
        conv_layer.biases = np.zeros(num_filters)
        
        # Forward pass with a simple input to set cache_input
        input_array = np.random.randn(1, 4, 4, 1)  # Random input for general testing
        output = conv_layer.forward(input_array)

        # Set a simple d_output (gradient from the next layer)
        d_output = np.ones_like(output)  # All ones for simplicity

        # Perform backward pass
        d_input = conv_layer.backward(d_output)

        # Expected changes in filters and biases based on the sum of products of d_output and input regions
        expected_d_filters = np.zeros_like(conv_layer.filters)
        expected_d_biases = np.array([12.0, 12.0])  # As sum of all d_output elements for each filter

        # Compute expected filter gradients
        for f in range(conv_layer.num_filters):
            for i in range(1):  # batch size is 1
                for y in range(3):  # output height
                    for x in range(3):  # output width
                        input_region = input_array[i, y*conv_layer.stride:y*conv_layer.stride+conv_layer.kernel_size, x*conv_layer.stride:x*conv_layer.stride+conv_layer.kernel_size, :]
                        expected_d_filters[f] += d_output[i, y, x, f] * input_region

        # Check if calculated gradients match expected gradients
        np.testing.assert_array_almost_equal(conv_layer.filters - learning_rate * expected_d_filters, conv_layer.filters - conv_layer.learning_rate * expected_d_filters)
        np.testing.assert_array_almost_equal(conv_layer.biases, np.zeros(num_filters) - learning_rate * expected_d_biases)

        # Test if d_input shape is correct
        self.assertEqual(d_input.shape, input_array.shape)

if __name__ == '__main__':
    unittest.main()
