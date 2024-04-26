import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TRAINING_DATA_PATH = "data/songs/training/"
VALIDATION_DATA_PATH = "data/songs/validation/"
SAVE_PATH = "data/"

SAMPLE_LENGTH = 10
SONG_START_OFFSET = 30


def load_data():
    '''
    Load songs and their dance style labels.

    Assumes you have a list of song paths and corresponding dance style labels.
    The folder name corresponds to the label (e.g., salsa, tango, ...)
    '''

    song_paths = glob.glob(f'{TRAINING_DATA_PATH}/*/*.mp3')
    dance_style_labels = [path.split('/')[-2] for path in song_paths]

    if len(song_paths) == 0:
        raise ValueError('No test songs found. Check the path.')

    print('Found', len(song_paths), 'songs.')
    print('Dance styles:', set(dance_style_labels))

    return song_paths, dance_style_labels


def split_songs(song_paths, dance_style_labels, length=SAMPLE_LENGTH):
    '''
    Loads the songs and splits them into 10-second segments.
    '''
    X = []
    y = []
    for path, label in zip(song_paths, dance_style_labels):
        song, sr = librosa.load(path, offset=SONG_START_OFFSET)
        for i in range(0, len(song) - sr*length, sr*length):
            X.append(song[i:i+sr*length])
            y.append(label)
    return np.array(X), np.array(y)


N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64


def create_spectrogram(waveform):
    '''
    Creates a spectrogram from the waveform.
    '''
    melspec = librosa.feature.melspectrogram(
        y=waveform, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)
    melspec = librosa.power_to_db(melspec**2)
    return melspec


def to_one_hot(labels, dimension):
    '''
    Converts labels to one-hot format.
    '''
    results = np.eye(dimension)[labels]
    return results


def create_test_data():
    '''
    Loads songs, splits them into 10-second segments, and creates spectrograms.
    Saves the training data to the file songs.npz
    '''

    song_paths, dance_style_labels = load_data()
    samples, sample_labels = split_songs(song_paths, dance_style_labels)
    spectrograms = np.array([create_spectrogram(sample) for sample in samples])

    # Handling labels
    label_encoder = LabelEncoder()
    dance_style_numbers = label_encoder.fit_transform(sample_labels)
    dance_style_one_hot = to_one_hot(dance_style_numbers, len(label_encoder.classes_))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        spectrograms, dance_style_one_hot, test_size=0.2, random_state=42)

    print('Train:', X_train.shape, y_train.shape)
    print('Test:', X_test.shape, y_test.shape)

    np.savez(f"{SAVE_PATH}/training.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             labels=label_encoder.classes_)


def create_validation_data():
    '''
    Loads validation data and creates spectrograms.
    Saves the validation data to the file validation.npz
    Only takes one sample per song in the validation data.
    '''

    validation_files = glob.glob(f'{VALIDATION_DATA_PATH}/*/*.mp3')
    validation_labels = [path.split('/')[-2] for path in validation_files]

    if len(validation_files) == 0:
        raise ValueError('No validation songs found. Check the path.')

    print('Found', len(validation_files), 'songs.')
    print('Dance styles:', set(validation_labels))

    def load_and_convert_audio(audio_file):
        waveform, sr = librosa.load(
            audio_file, offset=SONG_START_OFFSET, duration=SAMPLE_LENGTH)
        return waveform

    validation_spectrograms = []

    for path in validation_files:
        waveform = load_and_convert_audio(path)
        spectrogram = create_spectrogram(waveform)
        validation_spectrograms.append(spectrogram)

    validation_spectrograms = np.array(validation_spectrograms)

    label_encoder = LabelEncoder()
    validation_labels_numerical = label_encoder.fit_transform(
        validation_labels)
    validation_labels_one_hot = to_one_hot(
        validation_labels_numerical, len(label_encoder.classes_))

    print('Validation:', validation_spectrograms.shape,
          validation_labels_one_hot.shape)

    np.savez(f"{SAVE_PATH}/validation.npz", X_val=validation_spectrograms,
             y_val=validation_labels_one_hot, labels=label_encoder.classes_)


if __name__ == "__main__":
    print('Creating test data...')
    create_test_data()
    print('Creating validation data...')
    create_validation_data()
