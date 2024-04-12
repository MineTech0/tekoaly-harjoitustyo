import unittest
import numpy as np
from ..layers import Flatten  # Adjust the import path to match the location of your Flatten class

class TestFlattenLayer(unittest.TestCase):
    def setUp(self):
        self.flatten = Flatten()
        self.input_array = np.random.randn(5, 2, 3, 4)  # Example shape: (5, 2, 3, 4)

    def test_forward_pass(self):
        # Test the forward pass to ensure the output is correctly flattened
        output = self.flatten.forward(self.input_array)
        expected_output_shape = (5, 2 * 3 * 4)  # Flattened except for the first dimension (batch size)
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(np.array_equal(output, self.input_array.reshape(5, -1)))

    def test_backward_pass(self):
        # Test the backward pass to ensure the gradient is correctly reshaped to the original input shape
        output_gradient = np.random.randn(5, 2 * 3 * 4)  # Gradient shape matches the flattened output
        self.flatten.forward(self.input_array)  # Ensure input_shape is set
        backward_output = self.flatten.backward(output_gradient, None)
        self.assertEqual(backward_output.shape, self.input_array.shape)
        self.assertTrue(np.array_equal(backward_output, output_gradient.reshape(5, 2, 3, 4)))

if __name__ == '__main__':
    unittest.main()
