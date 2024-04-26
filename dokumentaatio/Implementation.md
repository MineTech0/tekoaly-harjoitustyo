# Implementation

## Idea

The idea was to creata neural network modal that cans to recognice dance style from music. This would be achived by training convolutional neural network to recognize the dance style from the music. The songs would be converted to spectrograms and the network would be trained to recognize the dance style from the spectrogram. The motivation for using convolutional neural network was that it is good at recognizing patterns from images and spectrogram is a image that represents the music in a way that is easy to analyze. I used music genre recognition as a reference for the project. 

References for music genre recognition:
- Thomas-Bouvier, “GitHub - thomas-bouvier/music-genre-recognition: Musical genre recognition using a CNN,” GitHub. https://github.com/thomas-bouvier/music-genre-recognition
- Priya-Dwivedi, “GitHub - priya-dwivedi/Music_Genre_Classification,” GitHub. https://github.com/priya-dwivedi/Music_Genre_Classification

## Structure

The project is divided to `own_model` and `peer_model`. The `peer_model` is used to compare the implementation of `own_model `which is the own implementation of neural network without using any neural network libary. There is also `preprocess_data.py` that is used to load and convert the song to spectograms.

## Preprocessing

[Preprocessing documentation](preprocessing.md)

## Peer model 

Peer model uses Tensorflow and Keras.
The model is a convolutional neural network that has 3 convolutional layers and 3 dense layers. It also uses batch normalization, max pooling and dropout layers.

```mermaid

graph TD;
    A["conv2d (Conv2D)"] --> B["batch_normalization (BatchNormalization)"]
    B --> C["max_pooling2d (MaxPooling2D)"]
    C --> D["conv2d_1 (Conv2D)"]
    D --> E["max_pooling2d_1 (MaxPooling2D)"]
    E --> F["dropout (Dropout)"]
    F --> G["conv2d_2 (Conv2D)"]
    G --> H["batch_normalization_1 (BatchNormalization)"]
    H --> I["max_pooling2d_2 (MaxPooling2D)"]
    I --> J["dropout_1 (Dropout)"]
    J --> K["flatten (Flatten)"]
    K --> L["dense (Dense)"]
    L --> M["dropout_2 (Dropout)"]
    M --> N["dense_1 (Dense)"]
    N --> O["dense_2 (Dense)"]
    O --> P["Output Layer"]


```
 ### Training

 The model is trained with 3 dance styles and and 33 songs from each style. The model is trained with 10 epochs.


## Own model

The own model is a convolutional neural network that has 3 convolutional layers and 3 dense layers. It also uses batch normalization, max pooling and dropout layers. It contains the same layers as the peer model but the implementation is done without using any neural network libary. The own model is very slow to train so it uses less filters and dense layer nodes than the peer model.

### Technical details

The model is implemented in `own_model` folder. The model is implemented in `neural_network.py` and the layers are implemented in `layers` folder. The layers are classes that implement the `BaseLayer` interface. 

Each layer has a `forward` and `backward` method that are used to calculate the forward and backward pass of the neural network. The `forward` method is used to calculate the output of the layer and the `backward` method is used to calculate the gradient of the layer. The `backward` method also updates the weights of the layer.

`initialize` method is used to initialize the weights of the layer. The weights are initialized with random values. And calculate the output shape of the layer. The `output_shape` is passed to the next layer when compiling the model.

The `NeuralNetwork` class is used to compile the model. Can be complied with `compile` method. The `compile` method is used to pass the input shape to the first layer and the output shape to the last layer. The `compile` method also initializes the weights of the layers. 

The `fit` method is used to train the model. The `fit` method uses the `forward` and `backward` methods of the layers to calculate the output and gradient of the model and updates the weights of the layers.
The `predict` method is used to predict the output of the model. The `predict` method uses the `forward` method of the layers to calculate the output of the model.

During training the model calculates the loss of the model with the [categorical_crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/) loss function and using the `softmax` activation function values form last layer. The loss gradien is clipped to prevent exploding gradients.

It is also possible to save and load the model with `save` and `load` methods.


