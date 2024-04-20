import unittest
import numpy as np
from ..layers import MaxPooling2D

class TestMaxPooling2D(unittest.TestCase):
    def test_forward(self):
        """
        Test the forward function of MaxPooling2D to ensure it correctly computes max pooling.
        """
        pool_size = 2
        stride = 2
        input_array = np.array([[[[1, 2], [3, 4]],
                                 [[5, 6], [7, 8]]]])  # 1x2x2x2 tensor

        expected_output = np.array([[[[7, 8]]]])  # 1x1x1x2 tensor after pooling

        maxpool = MaxPooling2D(pool_size, stride)
        maxpool.initialize(input_shape=(2, 2, 2))
        output = maxpool.forward(input_array)
        
        np.testing.assert_array_equal(output, expected_output)

    def test_backward(self):
        """
        Test the backward function of MaxPooling2D to ensure it correctly distributes gradients.
        """
        pool_size = 2
        stride = 2
        input_array = np.array([[[[1, 2], [3, 4]],
                                 [[5, 6], [7, 8]]]])  # 1x2x2x2 tensor
        output_gradient = np.array([[[[1, 1]]]])  # Gradient w.r.t output, 1x1x1x2 tensor

        maxpool = MaxPooling2D(pool_size, stride)
        maxpool.initialize(input_shape=(2, 2, 2))
        maxpool.forward(input_array)  # Set up forward pass to populate cache
        input_gradient = maxpool.backward(output_gradient, None)

        expected_input_gradient = np.array([[[[0, 0], [0, 1]],
                                     [[0, 0], [0, 1]]]])


        np.testing.assert_array_equal(input_gradient, expected_input_gradient)

if __name__ == "__main__":
    unittest.main()
