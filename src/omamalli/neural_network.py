import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output, training=False)
        return output

    def compute_loss(self, predicted_output, true_output):
        # This is a simplified mean squared error loss
        return np.mean((predicted_output - true_output) ** 2)

    def compute_loss_gradient(self, predicted_output, true_output):
        # Gradient of mean squared error loss w.r.t. predicted output
        return 2 * (predicted_output - true_output) / true_output.size

    def fit(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            predicted_output = self.predict(X_train)

            # Compute loss
            loss = self.compute_loss(predicted_output, Y_train)
            print(f"Epoch {epoch+1}, Loss: {loss}")

            # Backward pass
            loss_gradient = self.compute_loss_gradient(predicted_output, Y_train)

            # Reverse the layers for a proper backward pass
            for layer in reversed(self.layers):
                loss_gradient = layer.backward(loss_gradient, learning_rate)
