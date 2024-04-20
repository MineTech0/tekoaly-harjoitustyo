import unittest
import numpy as np
from ..layers import Dense

class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.input_size, self.output_size = 4, 3
        self.dense = Dense(self.output_size)
        self.dense.initialize(input_shape=(self.input_size,))
        self.input_array = np.random.randn(1, self.input_size)
        self.output_gradient = np.random.randn(1, self.output_size)
        self.learning_rate = 0.01

    def test_initialization(self):
        self.assertEqual(self.dense.weights.shape, (self.input_size, self.output_size))
        self.assertEqual(self.dense.biases.shape, (1, self.output_size))
        self.assertTrue(np.allclose(self.dense.weights, np.zeros_like(self.dense.weights), atol=0.1))
        self.assertTrue(np.all(self.dense.biases == np.zeros_like(self.dense.biases)))

    def test_forward_pass(self):
        output = self.dense.forward(self.input_array)
        expected_output = np.dot(self.input_array, self.dense.weights) + self.dense.biases
        self.assertEqual(output.shape, (1, self.output_size))
        self.assertTrue(np.allclose(output, expected_output))

    def test_backward_pass(self):
        self.dense.forward(self.input_array)
        input_gradient = self.dense.backward(self.output_gradient, self.learning_rate)
        expected_input_gradient = np.dot(self.output_gradient, self.dense.weights.T)
        self.assertEqual(input_gradient.shape, self.input_array.shape)
        self.assertTrue(np.allclose(input_gradient, expected_input_gradient))

    def test_parameter_updates(self):
        old_weights = self.dense.weights.copy()
        old_biases = self.dense.biases.copy()
        self.dense.forward(self.input_array)
        self.dense.backward(self.output_gradient, self.learning_rate)
        expected_new_weights = old_weights - self.learning_rate * np.dot(self.input_array.T, self.output_gradient)
        expected_new_biases = old_biases - self.learning_rate * np.sum(self.output_gradient, axis=0, keepdims=True)
        self.assertTrue(np.allclose(self.dense.weights, expected_new_weights))
        self.assertTrue(np.allclose(self.dense.biases, expected_new_biases))

if __name__ == '__main__':
    unittest.main()
