import unittest
import numpy as np
from ..layers import Dropout  # Adjust the import path to match the location of your Dropout class

class TestDropoutLayer(unittest.TestCase):
    def setUp(self):
        self.rate = 0.3
        self.dropout = Dropout(self.rate)
        self.input_array = np.random.randn(10, 10)  # Example input

    def test_initialization(self):
        # Check if the dropout rate is set correctly
        self.assertEqual(self.dropout.rate, self.rate)

    def test_forward_pass_training(self):
        # Test forward pass during training
        np.random.seed(0)  # For reproducibility
        output = self.dropout.forward(self.input_array, training=True)
        mask = self.dropout.mask
        expected_output = self.input_array * mask
        self.assertTrue(np.array_equal(output, expected_output))
        # Ensure some elements are zeroed out due to dropout
        self.assertTrue(np.sum(output == 0) > 0)

    def test_forward_pass_testing(self):
        # Test forward pass during testing (no dropout should be applied)
        output = self.dropout.forward(self.input_array, training=False)
        self.assertTrue(np.array_equal(output, self.input_array))

    def test_backward_pass(self):
        # Test the backward pass to ensure gradients are passed correctly
        output_gradient = np.random.randn(10, 10)
        np.random.seed(0)  # Ensure mask is the same as in forward pass training
        self.dropout.forward(self.input_array, training=True)
        backward_output = self.dropout.backward(output_gradient, learning_rate=None)
        expected_backward_output = output_gradient * self.dropout.mask
        self.assertTrue(np.array_equal(backward_output, expected_backward_output))

if __name__ == '__main__':
    unittest.main()
