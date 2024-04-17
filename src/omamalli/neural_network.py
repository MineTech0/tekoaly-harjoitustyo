import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self):
        # Initialize the weights of the network
        for i, layer in enumerate(self.layers):
            if i > 0 and hasattr(layer, 'initialize'):
                print(f"Initializing layer {layer.__class__} with input shape {self.layers[i-1].output_shape}.")
                layer.initialize(self.layers[i-1].output_shape)
            else:
                layer.initialize()
                print(f"Initializing layer {layer.__class__}. with input shape {layer.input_shape}")

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

    def fit(self, X_train, Y_train, epochs, learning_rate, batch_size, X_val=None, Y_val=None):
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"Starting training with {n_samples} samples, {n_batches} batches per epoch, and batch size {batch_size}.")

        for epoch in range(epochs):
            # Shuffling data randomly to avoid order bias
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            print(f"Epoch {epoch+1}/{epochs} started, data shuffled.")

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Forward propagation and loss computation for the mini-batch
                predicted_output = self.predict(X_batch)
                loss = self.compute_loss(predicted_output, Y_batch)

                # Backpropagation and weights update
                loss_gradient = self.compute_loss_gradient(predicted_output, Y_batch)
                for layer in reversed(self.layers):
                    loss_gradient = layer.backward(loss_gradient, learning_rate)

                print(f"Batch {i+1}/{n_batches}, Training Loss: {loss:.4f}")

            # Optional: Perform validation after training
            if X_val is not None and Y_val is not None:
                val_predicted_output = self.predict(X_val)
                val_loss = self.compute_loss(val_predicted_output, Y_val)
                print(f"Epoch {epoch+1}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}, Training Loss: {loss:.4f}")

        print("Training completed.")

