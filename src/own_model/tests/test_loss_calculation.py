import unittest
import numpy as np
from TestLayer import TestLayer
from own_model.neural_network import NeuralNetwork

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        # Set up a small neural network object which includes the loss functions
        self.nn = NeuralNetwork()

    def test_loss_correctness(self):
        predicted = np.array([[0.1, 0.9], [0.8, 0.2]])
        true = np.array([[0, 1], [1, 0]])
        loss = self.nn.compute_loss(predicted, true)
        # Manually computed expected loss
        expected_loss = - (0 * np.log(0.1) + 1 * np.log(0.9) + 1 * np.log(0.8) + 0 * np.log(0.2)) / 2
        np.testing.assert_almost_equal(loss, expected_loss)

    def test_compute_loss_gradient(self):
        """
        Test the compute_loss_gradient method.
        """
        neural_network = NeuralNetwork()

        # Test case 1: Single sample
        y_predicted = np.array([[0.2, 0.3, 0.5]])
        y_true = np.array([[0, 1, 0]])
        expected_gradient = np.array([[0.2, -0.7, 0.5]])
        computed_gradient = neural_network.compute_loss_gradient(y_predicted, y_true)
        np.testing.assert_array_almost_equal(computed_gradient, expected_gradient, decimal=6)


    def test_numerical_stability(self):
        predicted = np.array([[1e-10, 1.0 - 1e-10], [1e-10, 1.0 - 1e-10]])
        true = np.array([[0, 1], [1, 0]])
        # Test that no errors (like divide by zero) occur and values are clipped
        loss = self.nn.compute_loss(predicted, true)
        gradient = self.nn.compute_loss_gradient(predicted, true)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_zero_probabilities(self):
        predicted = np.array([[0, 1], [1, 0]])
        true = np.array([[0, 1], [1, 0]])
        # Ensure handling of zero probabilities without runtime errors
        loss = self.nn.compute_loss(predicted, true)
        gradient = self.nn.compute_loss_gradient(predicted, true)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_input_shape_errors(self):
        with self.assertRaises(ValueError):
            self.nn.compute_loss(np.array([1, 2, 3]), np.array([1, 2]))

        with self.assertRaises(ValueError):
            self.nn.compute_loss_gradient(np.array([1, 2, 3]), np.array([1, 2]))
if __name__ == '__main__':
    unittest.main()
