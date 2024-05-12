import tensorflow
import librosa
import os
from tensorflow.keras import backend as K
import numpy as np
import pickle
import sys

from ...own_model import neural_network
from ...own_model import layers
sys.modules['neural_network'] = neural_network
sys.modules['layers'] = layers

N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64

def metric(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

class PredictionService:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        self.peer_model = tensorflow.keras.models.load_model(os.path.join(current_directory, 'models/peer_model.keras'), custom_objects={'metric': metric})
        self.own_model = self.load(os.path.join(current_directory, 'models/own_model.pkl'))

    def preprocess_song(self, audio_file):
        waveform, sr = librosa.load(
            audio_file, offset=30, duration=10)
        return self.create_spectrogram(waveform)


    def create_spectrogram(self, waveform):
        '''
        Creates a spectrogram from the waveform.
        '''
        melspec = librosa.feature.melspectrogram(
            y=waveform, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)
        melspec = librosa.power_to_db(melspec**2)
        return melspec

    def predict_genre_peer_model(self, audio_file):
        '''
        Predicts the genre of a song using the peer model.
        '''
        processed_data = self.preprocess_song(audio_file)
        processed_data = processed_data.reshape(1, processed_data.shape[0], processed_data.shape[1], 1)
        prediction = self.peer_model.predict(processed_data)
        label, confidence = self.get_label(prediction[0])
        return label, confidence
    
    def predict_genre_own_model(self, audio_file):
        '''
        Predicts the genre of a song using the own model.
        '''
        processed_data = self.preprocess_song(audio_file)
        processed_data = processed_data.reshape(1, processed_data.shape[0], processed_data.shape[1], 1)
        prediction = self.own_model.predict(processed_data)
        label, confidence = self.get_label(prediction[0])
        return label, confidence
        
    
    def get_label(self, prediction):
        labels = ['fusku', 'salsa', 'valssi']
        #GET INDEX OF MAX VALUE
        index = np.argmax(prediction)
        return labels[index], prediction[index]
    
    def load(self, filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}.")
        return model
    
    
    