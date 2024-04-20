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

    def test_gradient_check(self):
        """Test gradient computation via numerical approximation"""
        np.random.seed(0)
        input_shape = (5, 5, 3)
        input_array = np.random.randn(1, *input_shape)
        num_filters = 2
        kernel_size = 3
        stride = 1
        
        conv_layer = Conv2D(num_filters, kernel_size, input_shape=input_shape, stride=stride)
        conv_layer.initialize(input_shape=input_shape)
        
        def fwd(input_array):
            return conv_layer.forward(input_array)
        
        output = fwd(input_array)
        output_grad = np.random.randn(*output.shape)

        # Compute the backward pass
        input_grad = conv_layer.backward(output_grad, learning_rate=0.01)

        # Numerical gradient checking
        epsilon = 1e-5
        it = np.nditer(input_array, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_value = input_array[ix]
            input_array[ix] = old_value + epsilon
            grad_approx = (fwd(input_array) - output) / epsilon
            input_array[ix] = old_value
            self.assertAlmostEqual(input_grad[ix], grad_approx[ix], places=5)
            it.iternext()

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

if __name__ == '__main__':
    unittest.main()
