# Testing

## Automated Testing

### Unit Testing

Unit tests rigorously examine our neural network implementation to ensure stability and functionality. These tests are crafted using the `unittest` module and reside within the `tests` directory. They are executed by running the command:

```bash
poetry run invoke test
```

These tests comprehensively verify both the initialization and operational correctness of neural network layers. Key aspects covered include:

- Proper initialization of network layers
- Correct functioning of forward and backward propagation methods
- Model compilation and training processes

Additionally, unit tests encompass checks for edge cases and common errors that might arise during the implementation phase.

### Code Coverage

Code coverage analysis is performed to evaluate the extent of codebase testing. The `coverage` module is utilized to generate a detailed report on the percentage of code covered by unit tests. The command to execute code coverage is:

```bash
poetry run invoke coverage
```

The coverage is now at 69%

## Model Evaluation

The performance of our custom-built neural network is benchmarked against a peer model constructed using TensorFlow and Keras. Both models undergo training with identical datasets to enable a fair comparison of accuracy metrics. Key evaluation points include:

- Comparison of model accuracies post-training

The own model and peer model doest contain the same layer or filter sizes, so the comparison is not completely fair. This is due to the performance limitations of the own model, which is implemented from scratch without the optimization features of TensorFlow and Keras.

## Manual Testing

Manual testing involves assessing the application's capability to recognize dance styles from music tracks. The web application is launched using the command:

```bash
poetry run invoke start
```

Once started, the application can be accessed at `http://localhost:5000`. Users can upload a song, and the application predicts the associated dance style. The results include:

- Dance style prediction from both the custom and peer model
- Confidence level of each prediction

This manual testing process is crucial for validating the practical application of the neural models in real-world scenarios. This uses pre trained models to predict the dance style of the uploaded song.