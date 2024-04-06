import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

KOULUTUS_DATA_POLKU = "data/kappaleet/koulutus/"
VALIDOINTI_DATA_POLKU = "data/kappaleet/validointi/"
TALLENNUS_POLKU = "data/"

NÄYTTEEN_PITUUS = 10
KAPPALEEN_ALKU_OFFSET = 30


def lataa_data():
    '''
    Lataa kappaleet ja niiden tanssilajilabelit.

    Oletetaan, että sinulla on lista kappalepolkuja ja vastaavia tanssilajilabeleita
    Kansion nimi vastaa labelia (esim. salsa, tango, ...) 
    '''

    kappalepolut = glob.glob(f'{KOULUTUS_DATA_POLKU}/*/*.mp3')
    tanssilajilabelit = [polku.split('/')[-2] for polku in kappalepolut]

    if len(kappalepolut) == 0:
        raise ValueError('Testi kappaleita ei löytynyt. Tarkista polku.')

    print('Löytyi', len(kappalepolut), 'kappaletta.')
    print('Tanssilajit:', set(tanssilajilabelit))

    return kappalepolut, tanssilajilabelit


def pilko_kappaleet(kappalepolut, tanssilajilabelit, pituus=NÄYTTEEN_PITUUS):
    '''
    Lataa kappaleet ja pilkkoo ne 10 sekunnin pituisiin pätkiin.
    '''
    X = []
    y = []
    for polku, label in zip(kappalepolut, tanssilajilabelit):
        kappale, sr = librosa.load(polku, offset=KAPPALEEN_ALKU_OFFSET)
        for i in range(0, len(kappale) - sr*pituus, sr*pituus):
            X.append(kappale[i:i+sr*pituus])
            y.append(label)
    return np.array(X), np.array(y)


N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64


def luo_spektrogrammi(aaltomuoto):
    '''
    Luo spektrogrammin aaltomuodosta.
    '''
    melspec = librosa.feature.melspectrogram(
        y=aaltomuoto, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)
    melspec = librosa.power_to_db(melspec**2)
    return melspec


def to_one_hot(labels, dimension):
    '''
    Muuntaa labelit one-hot muotoon.
    '''
    results = np.eye(dimension)[labels]
    return results


def luo_testidata():
    '''
    Lataa kappaleet, pilkkoo ne 10 sekunnin pituisiin pätkiin ja luo spektrogrammit.
    Tallentaa koulutusdatan tiedostoon kappaleet.npz
    '''

    kappalepolut, tanssilajilabelit = lataa_data()
    näytteet, näytteiden_labelit = pilko_kappaleet(
        kappalepolut, tanssilajilabelit)
    spektrogrammit = np.array([luo_spektrogrammi(näyte) for näyte in näytteet])

    # Labelien käsittely
    label_encoder = LabelEncoder()
    tanssilaji_numerot = label_encoder.fit_transform(näytteiden_labelit)
    tanssilaji_one_hot = to_one_hot(
        tanssilaji_numerot, len(label_encoder.classes_))

    # Jaa data opetus- ja testisetteihin
    X_train, X_test, y_train, y_test = train_test_split(
        spektrogrammit, tanssilaji_one_hot, test_size=0.2, random_state=42)

    print('Train:', X_train.shape, y_train.shape)
    print('Test', X_test.shape, y_test.shape)

    np.savez(f"{TALLENNUS_POLKU}/kappaleet.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             labels=label_encoder.classes_)


def luo_validointi_data():
    '''
    Lataa validointidatan ja luo spektrogrammit.
    Tallentaa validointidatan tiedostoon validointi.npz
    Validointi datassa otetaan vain yksi näyte per kappale.
    '''

    validation_files = glob.glob(f'{VALIDOINTI_DATA_POLKU}/*/*.mp3')
    validation_labels = [polku.split('/')[-2] for polku in validation_files]

    if len(validation_files) == 0:
        raise ValueError('Validointi kappaleita ei löytynyt. Tarkista polku.')

    print('Löytyi', len(validation_files), 'kappaletta.')
    print('Tanssilajit:', set(validation_labels))

    def lataa_ja_muunna_audio(audio_tiedosto):
        aaltomuoto, sr = librosa.load(
            audio_tiedosto, offset=KAPPALEEN_ALKU_OFFSET, duration=NÄYTTEEN_PITUUS)
        return aaltomuoto

    validation_spectrograms = []

    for polku in validation_files:
        aaltomuoto = lataa_ja_muunna_audio(polku)
        spektrogrammi = luo_spektrogrammi(aaltomuoto)
        validation_spectrograms.append(spektrogrammi)

    validation_spectrograms = np.array(validation_spectrograms)

    label_encoder = LabelEncoder()
    validation_labels_numerical = label_encoder.fit_transform(
        validation_labels)
    validation_labels_one_hot = to_one_hot(
        validation_labels_numerical, len(label_encoder.classes_))

    print('Validation:', validation_spectrograms.shape,
          validation_labels_one_hot.shape)

    np.savez(f"{TALLENNUS_POLKU}/validointi.npz", X_val=validation_spectrograms,
             y_val=validation_labels_one_hot, labels=label_encoder.classes_)


if __name__ == "__main__":
    print('Luodaan testidata...')
    luo_testidata()
    print('Luodaan validointidata...')
    luo_validointi_data()
