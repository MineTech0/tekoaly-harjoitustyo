import unittest
import numpy as np
from omamalli.neural_network import NeuralNetwork
from .TestLayer import TestLayer

class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()

    def test_add_layer(self):
        layer = TestLayer()
        self.nn.add(layer)
        self.assertIn(layer, self.nn.layers)

    def test_compile_layers(self):
        first_layer = TestLayer()
        second_layer = TestLayer()
        self.nn.add(first_layer)
        self.nn.add(second_layer)
        self.nn.compile()
        self.assertEqual(second_layer.input_shape, 100)

    def test_predict(self):
        layer = TestLayer()
        self.nn.add(layer)
        self.nn.compile()
        result = self.nn.predict(np.zeros(10))
        self.assertEqual(result.shape[0], 100)

    def test_compute_loss(self):
        predicted = np.array([1, 2, 3])
        true = np.array([1, 2, 3])
        loss = self.nn.compute_loss(predicted, true)
        self.assertEqual(loss, 0)

    def test_compute_loss_gradient(self):
        predicted = np.array([1, 2, 3])
        true = np.array([1, 2, 3])
        gradient = self.nn.compute_loss_gradient(predicted, true)
        self.assertTrue(np.array_equal(gradient, np.zeros_like(predicted)))

    def test_training(self):
        layer = TestLayer()
        self.nn.add(layer)
        self.nn.compile()
        X_train = np.zeros((10, 10))
        Y_train = np.zeros((10, 100))
        self.nn.fit(X_train, Y_train, epochs=1, learning_rate=0.01, batch_size=5)
        # Check if training does not raise any exceptions

    def test_save_load(self):
        import os
        filename = 'test_model.pkl'
        self.nn.save(filename)
        self.assertTrue(os.path.exists(filename))
        loaded_model = NeuralNetwork.load(filename)
        self.assertIsInstance(loaded_model, NeuralNetwork)
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
