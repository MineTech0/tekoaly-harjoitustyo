import pickle
import numpy as np
import time

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

    def predict(self, input_data, training=False):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def accuracy(self, predicted_output, true_output):
        predicted_labels = np.argmax(predicted_output, axis=1)
        true_labels = np.argmax(true_output, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def compute_loss(self, predicted_output, true_output, epsilon=1e-8):
        # Clip predictions to avoid log(0) and division by zero issues
        predicted_output = np.clip(predicted_output, epsilon, 1. - epsilon)
        # Compute cross-entropy from probabilities
        loss = -np.sum(true_output * np.log(predicted_output), axis=1)
        # Average over the batch
        return np.mean(loss)

    def compute_loss_gradient(self, predicted_output, true_output, epsilon=1e-8):
        # Clip predictions to ensure stability in division and log operations
        predicted_output = np.clip(predicted_output, epsilon, 1. - epsilon)
        # Compute gradient of the loss
        gradient = -(true_output / predicted_output) / predicted_output.shape[0]
        return gradient


    def fit(self, X_train, Y_train, epochs, learning_rate, batch_size, X_val=None, Y_val=None, save_best_only=True, filename='best_model.pkl'):
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        best_loss = float('inf')
        clip_norm = 1.0
        total_batch_times = []

        print(f"Starting training with {n_samples} samples, {n_batches} batches per epoch, and batch size {batch_size}.")

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            print(f"Epoch {epoch+1}/{epochs} started, data shuffled.")

            epoch_loss = 0
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]
                start_time = time.time()

                predicted_output = self.predict(X_batch, training=True)
                loss = self.compute_loss(predicted_output, Y_batch)
                epoch_loss += loss
                loss_gradient = self.compute_loss_gradient(predicted_output, Y_batch)
                for layer in reversed(self.layers):
                    gradient_norm = np.linalg.norm(loss_gradient)
                    if gradient_norm > clip_norm:
                        loss_gradient *= clip_norm / gradient_norm
                    loss_gradient = layer.backward(loss_gradient, learning_rate)

                end_time = time.time()
                batch_time = end_time - start_time
                total_batch_times.append(batch_time)
                accuracy = self.accuracy(predicted_output, Y_batch)

                if len(total_batch_times) > 0:
                    average_batch_time_so_far = sum(total_batch_times) / len(total_batch_times)
                    remaining_batches = (n_batches * epochs) - (epoch * n_batches + i + 1)
                    estimated_remaining_time = average_batch_time_so_far * remaining_batches
                    estimated_remaining_time_minutes = estimated_remaining_time / 60
                    print(f"Batch {i+1}/{n_batches}, Epoch {epoch+1}/{epochs}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {batch_time:.4f} seconds")
                    print(f"Estimated remaining training time: {estimated_remaining_time_minutes:.2f} seconds")

            epoch_loss /= n_batches
            print(f"Epoch {epoch+1} completed with average training loss {epoch_loss:.4f}.")

            if X_val is not None:
                val_predicted_output = self.predict(X_val)
                val_loss = self.compute_loss(val_predicted_output, Y_val)
                print(f"Validation Loss: {val_loss:.4f}")
                if save_best_only and val_loss < best_loss:
                    best_loss = val_loss
                    self.save(filename)
            else:
                if save_best_only and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save(filename)

        print("Training completed.")

        
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filename}.")

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}.")
        return model

