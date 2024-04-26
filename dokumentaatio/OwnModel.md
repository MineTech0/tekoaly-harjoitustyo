# Neural network implementation

The neural network is implemented in `own_model` folder. The model is implemented in `neural_network.py` and the layers are implemented in `layers` folder. The layers are classes that implement the `BaseLayer` interface.

## Neural network

The neural network is implemented in `neural_network.py`. The neural network is a class that is used to compile and train the model.

### Building the model

The model is built with `add` method. The `add` method is used to add layers to the model. The `add` method takes a layer as a parameter and adds the layer to the model. The first layer needs the shape of the input data and the last layer needs the shape of the output data.

### Compiling the model

The model is compiled with `compile` method. Compile method calls `initialize` method of the layers to initialize the weights of the layers. The `initialize` implemented by all layers calculates the output shape of the layer and retuns it to the next layer. Now all layers have input and output shape.

### Training the model

The model is trained with `fit` method. The `fit` method uses the `forward` and `backward` methods of the layers to calculate the output and gradient of the model and updates the weights of the layers.

The fit method needs the input data, target data, number of epochs, learning rate, batch size and validation data. The network splits the input data to batches and trains the model with each batch. This is called [mini-batch gradient descent](https://medium.com/@dancerworld60/mini-batch-gradient-descent-1c36b8103f2c). The model is trained with the input data and target data for the number of epochs. The model is validated with the validation data after each epoch.

After each epoch the model is saved if the validation loss is lower than the previous validation loss. The model is saved with `save` method. The `save` method saves the model class to pickle file. This way the model can be loaded and used later.

#### Loss function

During training the model calculates the loss of the model with the [categorical_crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/) loss function and using the `softmax` activation function values form last layer. The loss gradien is clipped to prevent exploding gradients. The loss is calculated with the `calculate_loss` method.

#### Calculating loss gradient

The first loss gradient is calculated with the `calculate_loss_gradient` method. The true label values are used to calculate the loss gradient which is passed back to last dense layer to update the weights of the model. The gradient is calculated in each layer with the `backward` method and passed to the previous layer.

### Predicting the output

The model can predict the output with `predict` method. The `predict` method uses the `forward` method of the layers to calculate the output of the model.

### Saving and loading the model

It is also possible to save and load the model with `save` and `load` methods. The `save` method saves the model class to pickle file. The `load` method loads the model class from pickle file.