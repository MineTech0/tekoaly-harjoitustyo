import unittest
import numpy as np
from ..layers import Softmax

class TestSoftmax(unittest.TestCase):
    def test_forward_single_vector(self):
        """
        Test the forward pass with a single vector.
        """
        softmax = Softmax()
        input_array = np.array([1.0, 2.0, 3.0])
        output = softmax.forward(input_array)

        # Calculating expected output using softmax formula
        exps = np.exp(input_array - np.max(input_array))
        expected_output = exps / np.sum(exps)
        
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

    def test_forward_batch_vectors(self):
        """
        Test the forward pass with a batch of vectors.
        """
        softmax = Softmax()
        input_array = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        output = softmax.forward(input_array)
        
        # Each row in the input should be independently normalized
        exps = np.exp(input_array - np.max(input_array, axis=-1, keepdims=True))
        expected_output = exps / np.sum(exps, axis=-1, keepdims=True)
        
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

    def test_backward_pass_through(self):
        """
        Test the backward function to ensure it simply passes the gradient unchanged.
        """
        softmax = Softmax()
        input_array = np.array([1.0, 2.0, 3.0])
        output_gradient = np.array([0.1, 0.2, 0.3])
        softmax.forward(input_array)  # Set the layer's state
        input_gradient = softmax.backward(output_gradient, None)

        np.testing.assert_array_equal(input_gradient, output_gradient)

if __name__ == "__main__":
    unittest.main()
