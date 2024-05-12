import pickle
import numpy as np
import time

class NeuralNetwork:
    """
    A simple neural network class that allows adding layers, compiling, training, 
    predicting, and evaluating the model's performance.
    """

    def __init__(self):
        """Initialize the neural network with an empty list of layers."""
        self.layers = []

    def add(self, layer):
        """
        Add a layer to the neural network.

        Args:
            layer (Layer): The layer to be added.
        """
        self.layers.append(layer)

    def compile(self):
        """
        Initialize the weights of the network for each layer based on the output
        shape of the preceding layer.
        """
        for i, layer in enumerate(self.layers):  # Iterate through all layers
            if i > 0 and hasattr(layer, 'initialize'):
                # If the layer is not the first one and can be initialized with an input shape
                print(f"Initializing layer {layer.__class__} with input shape {self.layers[i-1].output_shape}.")
                layer.initialize(self.layers[i-1].output_shape)  # Initialize using shape of the previous layer
            else:
                print(f"Initializing layer {layer.__class__}.")
                layer.initialize()  # Initialize the first layer

    def predict(self, input_data, training=False):
        """
        Compute the output of the network for the given input data.

        Args:
            input_data (np.array): The input data.
            training (bool): Flag to indicate whether the model is in training mode.

        Returns:
            np.array: The output of the network.
        """
        output = input_data  # Start with the input data
        for layer in self.layers:  # Loop through each layer
            output = layer.forward(output, training=training)  # Pass output through the current layer
        return output  # Return the final output
    
    def accuracy(self, predicted_output, true_output):
        """
        Calculate the accuracy of the predictions.

        Args:
            predicted_output (np.array): The predicted outputs.
            true_output (np.array): The true outputs.

        Returns:
            float: The accuracy metric.
        """
        predicted_labels = np.argmax(predicted_output, axis=1)  # Find the index of max value in predictions
        true_labels = np.argmax(true_output, axis=1)  # Find the index of max value in true labels
        return np.mean(predicted_labels == true_labels)  # Calculate and return the mean accuracy

    def compute_loss(self, predicted_output, true_output, epsilon=1e-8):
        """
        Compute the cross-entropy loss between the predicted and true outputs.

        Args:
            predicted_output (np.array): The predicted probabilities.
            true_output (np.array): The true probabilities.
            epsilon (float): Small number to prevent division by zero.

        Returns:
            float: The average loss.
        """
        predicted_output = np.clip(predicted_output, epsilon, 1. - epsilon)  # Clip predictions to avoid log(0)
        loss = -np.sum(true_output * np.log(predicted_output), axis=1)  # Compute the cross-entropy loss
        return np.mean(loss)  # Return the average loss over all samples

    def compute_loss_gradient(self, y_predicted, y_true):
        """
        Compute the gradient of the loss with respect to the output.

        Args:
            y_predicted (np.array): Predicted outputs.
            y_true (np.array): True outputs.

        Returns:
            np.array: Gradient of the loss.
        """
        return (y_predicted - y_true) / y_true.shape[0]  # Calculate and return the gradient normalized by batch size

    def fit(self, X_train, Y_train, epochs, learning_rate, batch_size, X_val=None, Y_val=None, save_best_only=True, filename='best_model.pkl'):
        """
        Train the neural network using the given training data.

        Args:
            X_train (np.array): Input features for training.
            Y_train (np.array): Target output for training.
            epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for optimization.
            batch_size (int): Number of samples per batch.
            X_val (np.array, optional): Input features for validation.
            Y_val (np.array, optional): Target output for validation.
            save_best_only (bool): Flag to save only the best model.
            filename (str): Base filename for saving the model.
        """
        n_samples = X_train.shape[0]  # Total number of samples in the training data
        n_batches = int(np.ceil(n_samples / batch_size))  # Number of batches needed per epoch
        best_loss = float('inf')  # Initialize the best loss as infinity for comparison
        total_batch_times = []  # List to store time taken for each batch

        print(f"Starting training with {n_samples} samples, {n_batches} batches per epoch.")

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)  # Shuffle indices to randomize the data order
            X_train = X_train[indices]  # Apply shuffled indices to the training data
            Y_train = Y_train[indices]
            print(f"Epoch {epoch+1}/{epochs} started, data shuffled.")

            epoch_loss = 0  # Reset epoch loss
            for i in range(n_batches):
                start, end = i * batch_size, (i + 1) * batch_size  # Calculate start and end indices for the batch
                X_batch, Y_batch = X_train[start:end], Y_train[start:end]  # Extract batch data
                start_time = time.time()  # Start time for batch processing

                predicted_output = self.predict(X_batch, training=True)  # Get predictions for the batch
                loss = self.compute_loss(predicted_output, Y_batch)  # Calculate loss for the batch
                epoch_loss += loss  # Accumulate total loss for the epoch
                loss_gradient = self.compute_loss_gradient(predicted_output, Y_batch)  # Calculate loss gradient

                for layer in reversed(self.layers):
                    loss_gradient = layer.backward(loss_gradient, learning_rate)  # Apply backpropagation

                batch_time = time.time() - start_time  # Time taken for processing the batch
                total_batch_times.append(batch_time)  # Store batch time
                accuracy = self.accuracy(predicted_output, Y_batch)  # Calculate accuracy for the batch

                if len(total_batch_times) > 0:
                    average_batch_time = sum(total_batch_times) / len(total_batch_times)  # Average batch time so far
                    remaining_batches = (n_batches * epochs) - (epoch * n_batches + i + 1)  # Remaining batches to process
                    estimated_remaining_time = average_batch_time * remaining_batches  # Estimated remaining time in seconds
                    print(f"Batch {i+1}/{n_batches}, Epoch {epoch+1}/{epochs}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {batch_time:.4f} seconds")
                    print(f"Estimated remaining training time: {estimated_remaining_time / 60:.2f} minutes.")

            epoch_loss /= n_batches  # Calculate average loss for the epoch
            print(f"Epoch {epoch+1} completed with average loss {epoch_loss:.4f}.")

            if X_val is not None and Y_val is not None:
                val_predicted_output = self.predict(X_val)  # Get predictions for validation data
                val_loss = self.compute_loss(val_predicted_output, Y_val)  # Calculate validation loss
                val_accuracy = self.accuracy(val_predicted_output, Y_val)  # Calculate validation accuracy
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                if save_best_only and val_loss < best_loss:
                    best_loss = val_loss  # Update best loss if the current one is better
                    self.save(f"{filename}_val_best_loss_{val_loss:.4f}_epoch_{epoch+1}.pkl")  # Save the model
            elif save_best_only and epoch_loss < best_loss:
                best_loss = epoch_loss  # Update best loss for training if the current one is better
                self.save(f"{filename}_train_best_loss_{epoch_loss:.4f}_epoch_{epoch+1}.pkl")  # Save the model

        print("Training completed.")

    def save(self, filename):
        """
        Save the current state of the neural network to a file.

        Args:
            filename (str): The filename to save the model.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)  # Use pickle to save the entire neural network object
        print(f"Model saved to {filename}.")

    @staticmethod
    def load(filename):
        """
        Load a neural network model from a file.

        Args:
            filename (str): The filename of the model to load.

        Returns:
            NeuralNetwork: The loaded neural network.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)  # Load the model from the pickle file
        print(f"Model loaded from {filename}.")
        return model
