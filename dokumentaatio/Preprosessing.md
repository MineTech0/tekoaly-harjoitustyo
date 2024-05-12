# Preprocessing

## Data Structure

The `data` directory contains the song files organized for model training and evaluation. Within the `data` folder, there are two subdirectories: `training` and `validation`. Each of these subdirectories contains further folders named after the dance styles they represent, which serve as labels for the songs. All songs are stored in `.mp3` format.

## Data Preprocessing

Data preprocessing is facilitated by the script `preprocess_data.py`. The steps involved in preprocessing are designed to convert raw audio files into a format suitable for machine learning models. The steps include:

1. **Load the Data**: Songs are loaded from the `data` directory, maintaining the organization by training and validation sets.
2. **Convert the Songs to Mel-Spectrograms**: The audio files are transformed into mel-spectrograms, which are a visually informative representation of the spectrum of frequencies in the audio as it varies with time.
3. **Save the Spectrograms**: The resulting spectrograms are saved in `.npz` format for efficiency and ease of access during model training.

The spectrograms are organized and stored in the `data` directory under the names `training.npz` and `validation.npz`, reflecting their respective datasets.

## Song Processing

The preprocessing of each song includes several specific operations aimed at standardizing the input data for consistent model training:

- **Initial Segment Skipping**: The first 30 seconds of each song are skipped to avoid introductory segments that might not be representative of the main content.
- **Song Splitting**: Each song is divided into 10-second clips, providing a uniform length for processing and analysis.
- **Sampling and Conversion**: Every song is loaded and sampled at a rate of 22050 Hz. The audio clips are then converted into mel-spectrograms using parameters:
  - `n_fft=512`: The length of the windowed signal after padding with zeros.
  - `hop_length=n_fft // 2`: The number of samples between successive frames.
  - `n_mels=64`: The number of Mel bands to generate.

- **Spectrogram Scaling**: The amplitude of the spectrograms is squared and then converted to decibel (dB) units to normalize the dynamic range.

## Training and Validation Data

The organization and usage of data are critical for effective model training and validation:

- **Training Data**: Used to train the neural network, constantly updating the weights of the model based on the computed loss during each iteration.

- **Testing Split**: To prevent overfitting, 20% of the training data is set aside as testing data. Employed to evaluate the model’s performance periodically during training. This dataset helps monitor for overfitting and tune the model's hyperparameters. 

- **Validation Data**: This data is used to evaluate the performance after each epoch and and after training. This helps ensure that the model generalizes well to new, unseen data. Unlike the training data, validation data consists solely of 10-second song clips and contains no overlapping songs with the training dataset.


## Labels

Labels for the songs are directly derived from the names of the folders within the `training` and `validation` directories. These labels are one-hot encoded, a common format for categorical variables, and stored in the `.npz` files alongside the spectrograms. This encoding transforms the label for each song into a binary vector representing the presence or absence of each category (dance style) within the dataset. This vector format is crucial for the classification tasks performed by the model.