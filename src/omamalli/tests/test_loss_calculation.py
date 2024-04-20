import unittest
import numpy as np
from .TestLayer import TestLayer
from omamalli.neural_network import NeuralNetwork

# Assuming the NeuralNetwork and TestLayer classes are already defined

class NeuralNetworkNaNTests(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        self.layer = TestLayer()
        self.nn.add(self.layer)
        self.nn.compile()

    def test_loss_computation_for_nan(self):
        # Generate input data that could lead to large values during forward propagation
        X_train = np.full((10, 10), 1e10)  # Very large input values
        Y_train = np.full((10, 100), 1e10)  # Very large target values

        # Run prediction and loss computation
        predicted_output = self.nn.predict(X_train, training=True)
        loss = self.nn.compute_loss(predicted_output, Y_train)
        loss_gradient = self.nn.compute_loss_gradient(predicted_output, Y_train)

        # Check if any NaN values appear in loss or gradient
        self.assertFalse(np.isnan(loss), "Loss computation resulted in NaN")
        self.assertFalse(np.any(np.isnan(loss_gradient)), "Loss gradient computation resulted in NaN")

    def test_high_learning_rate_for_nan(self):
        # Using a high learning rate that might cause instability
        high_learning_rate = 1e3  # Unusually high learning rate
        X_train = np.random.normal(size=(10, 10))
        Y_train = np.random.normal(size=(10, 100))
        self.nn.fit(X_train, Y_train, epochs=1, learning_rate=high_learning_rate, batch_size=5)

        # Directly accessing the last computed loss to see if it's NaN
        predicted_output = self.nn.predict(X_train, training=True)
        loss = self.nn.compute_loss(predicted_output, Y_train)
        self.assertFalse(np.isnan(loss), "Loss computation resulted in NaN with high learning rate")

if __name__ == '__main__':
    unittest.main()
