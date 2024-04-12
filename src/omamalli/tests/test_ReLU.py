import unittest
import numpy as np
from ..layers import ReLU

class TestReLU(unittest.TestCase):
    def test_forward_positive_input(self):
        """
        Test the forward pass with positive input values.
        """
        relu = ReLU()
        input_array = np.array([1, 2, 3])
        output = relu.forward(input_array)
        np.testing.assert_array_equal(output, input_array)

    def test_forward_negative_input(self):
        """
        Test the forward pass with negative input values.
        """
        relu = ReLU()
        input_array = np.array([-1, -2, -3])
        output = relu.forward(input_array)
        np.testing.assert_array_equal(output, np.zeros_like(input_array))

    def test_forward_mixed_input(self):
        """
        Test the forward pass with a mix of positive and negative input values.
        """
        relu = ReLU()
        input_array = np.array([-1, 0, 1])
        expected_output = np.array([0, 0, 1])
        output = relu.forward(input_array)
        np.testing.assert_array_equal(output, expected_output)

    def test_backward_positive_input(self):
        """
        Test the backward pass with positive input values.
        """
        relu = ReLU()
        input_array = np.array([1, 2, 3])
        output_gradient = np.array([1, 1, 1])
        relu.forward(input_array)  # Must call forward to set `self.input`
        input_gradient = relu.backward(output_gradient, None)
        np.testing.assert_array_equal(input_gradient, output_gradient)

    def test_backward_negative_input(self):
        """
        Test the backward pass with negative input values.
        """
        relu = ReLU()
        input_array = np.array([-1, -2, -3])
        output_gradient = np.array([1, 1, 1])
        relu.forward(input_array)
        input_gradient = relu.backward(output_gradient, None)
        np.testing.assert_array_equal(input_gradient, np.zeros_like(input_array))

    def test_backward_mixed_input(self):
        """
        Test the backward pass with a mix of positive and negative input values.
        """
        relu = ReLU()
        input_array = np.array([-1, 0, 1])
        output_gradient = np.array([1, 1, 1])
        relu.forward(input_array)
        input_gradient = relu.backward(output_gradient, None)
        expected_gradient = np.array([0, 0, 1])
        np.testing.assert_array_equal(input_gradient, expected_gradient)

if __name__ == "__main__":
    unittest.main()
