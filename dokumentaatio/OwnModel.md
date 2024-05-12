# Neural Network Implementation

The neural network is structured within the `own_model` directory. The core functionality of the model is contained in the `neural_network.py` file, while individual layers are defined within the `layers` directory. These layers are designed as classes, all inheriting from the `BaseLayer` interface, ensuring a standardized approach across different types of layers.

## Neural Network Architecture

### Overview
The primary functionality of the neural network is encapsulated within the `NeuralNetwork` class located in `neural_network.py`. This class serves as the main interface for compiling and training the model, managing the flow of data through the network's layers.

### Building the Model
The model construction is facilitated by the `add` method of the `NeuralNetwork` class. This method allows for the sequential addition of layers to the network:

- **Layer Addition**: The `add` method accepts a layer instance as a parameter, appending it to the model's internal list of layers.
- **Input and Output Shapes**: The first layer added to the model must specify the shape of the input data, while the final layer should define the shape of the output data to match the expected results.

### Compiling the Model
Compiling the model involves preparing it for training by initializing the weights of each layer:

- **Initialization**: The `compile` method invokes the `initialize` method on each layer, which calculates and returns the output shape to be used by subsequent layers.
- **Weight Initialization**: During this phase, layers adjust their internal parameters (weights and biases) based on the input shape they receive.

### Training the Model
The training process is conducted through the `fit` method, which implements the [mini-batch gradient descent](https://medium.com/@dancerworld60/mini-batch-gradient-descent-1c36b8103f2c) algorithm:

- **Batch Processing**: The input data is divided into batches, with each batch used to train the model in iterations (epochs).
- **Epochs and Learning Parameters**: The method requires specifications of the number of epochs, learning rate, batch size, and validation data.
- **Validation and Model Saving**: After training on each epoch, the model is validated against the provided validation data. If the validation loss improves, the model state is saved using the `save` method.

#### Loss Function
- **Loss Calculation**: The model uses the [categorical cross-entropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/) loss function, with gradients being clipped to avoid exploding gradients.
- **Gradient Calculation**: Loss gradients are computed starting from the output layer, utilizing the `calculate_loss_gradient` method, and propagated back through the network.

### Predicting the Output
Output predictions are made using the `predict` method:

- **Forward Propagation**: This method leverages the `forward` methods of each layer to compute and output the final predictions of the network.

### Saving and Loading the Model
The model's state can be preserved and retrieved using the `save` and `load` methods:

- **Saving**: The `save` method serializes the entire model class to a pickle file, allowing for long-term storage.
- **Loading**: The `load` method deserializes the model from the pickle file, restoring its state for further use or additional training.

This documentation provides a detailed guide to implementing, compiling, training, and utilizing a neural network within a structured Python project.