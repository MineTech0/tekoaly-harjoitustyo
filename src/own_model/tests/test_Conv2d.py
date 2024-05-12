import numpy as np
import unittest
from own_model.layers import Conv2D

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

    def test_gradient_shapes(self):
        """Test that the shapes of gradients are correct"""
        input_shape = (5, 5, 3)
        input_array = np.random.randn(1, *input_shape)
        num_filters = 2
        kernel_size = 3
        stride = 1
        
        conv_layer = Conv2D(num_filters, kernel_size, input_shape=input_shape, stride=stride)
        conv_layer.initialize(input_shape=input_shape)
        output = conv_layer.forward(input_array)
        output_grad = np.random.randn(*output.shape)
        input_grad = conv_layer.backward(output_grad, learning_rate=0.01)
        
        self.assertEqual(conv_layer.filters.shape, (num_filters, kernel_size, kernel_size, input_shape[2]))
        self.assertEqual(input_grad.shape, input_array.shape)
        self.assertEqual(len(conv_layer.biases), num_filters)

    def test_weights_bias_update(self):
        """Test weights and biases update"""
        input_shape = (5, 5, 3)
        input_array = np.random.randn(1, *input_shape)
        num_filters = 2
        kernel_size = 3
        stride = 1
        
        conv_layer = Conv2D(num_filters, kernel_size, input_shape=input_shape, stride=stride)
        conv_layer.initialize(input_shape=input_shape)
        
        initial_filters = np.copy(conv_layer.filters)
        initial_biases = np.copy(conv_layer.biases)
        
        output = conv_layer.forward(input_array)
        output_grad = np.random.randn(*output.shape)
        conv_layer.backward(output_grad, learning_rate=0.01)
        
        self.assertFalse(np.allclose(initial_filters, conv_layer.filters))
        self.assertFalse(np.allclose(initial_biases, conv_layer.biases))
        
    def test_no_padding_perfect_fit(self):
        """ Test convolution with no padding where the kernel fits the input dimensions perfectly. """
        input_shape = (8, 8, 3)
        num_filters = 2
        kernel_size = 3
        stride = 1
        conv_layer = Conv2D(num_filters, kernel_size, input_shape, stride)
        input_array = np.random.rand(1, 8, 8, 3)  # Batch size 1
        output = conv_layer.forward(input_array)
        expected_output_shape = (6, 6, num_filters)  # (8-3)//1 + 1 = 6
        self.assertEqual(output.shape[1:], expected_output_shape)

    def test_no_padding_stride(self):
        """ Test convolution with no padding and a stride greater than 1. """
        input_shape = (8, 8, 3)
        num_filters = 2
        kernel_size = 3
        stride = 2
        conv_layer = Conv2D(num_filters, kernel_size, input_shape, stride)
        input_array = np.random.rand(1, 8, 8, 3)
        output = conv_layer.forward(input_array)
        expected_output_shape = (3, 3, num_filters)  # (8-3)//2 + 1 = 3
        self.assertEqual(output.shape[1:], expected_output_shape)

    def test_input_shape_mismatch(self):
        """ Test error handling when the input shape does not match the expected shape. """
        input_shape = (8, 8, 3)
        num_filters = 2
        kernel_size = 3
        stride = 1
        conv_layer = Conv2D(num_filters, kernel_size, input_shape, stride)
        wrong_input_array = np.random.rand(1, 7, 7, 3)  # Incorrect dimensions
        with self.assertRaises(ValueError):
            conv_layer.forward(wrong_input_array)
if __name__ == '__main__':
    unittest.main()
