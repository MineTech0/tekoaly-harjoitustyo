import numpy as np

from neural_network import NeuralNetwork
from layers import Dense, ReLU, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax, BatchNormalization
import pathlib

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "training.npz" 

def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)

    # Accessing the training and testing data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    labels = data['labels']
    
    return X_train, y_train, X_test, y_test, labels



X_train, y_train, X_test, y_test, labels = load_data()

#add channel to the data
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("labels:", labels)

model = NeuralNetwork()

model.add(Conv2D(16, 5, input_shape=(X_train.shape[1], X_train.shape[2], 1), l2_lambda=0.04))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Conv2D(8, 3, l2_lambda=0.04))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, l2_lambda=0.04))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(Dense(32, l2_lambda=0.04 ))
model.add(ReLU())
model.add(Dense(y_train.shape[1]))
model.add(Softmax())

model.compile()
 
model.fit(X_train, y_train, epochs=10, learning_rate=0.01, batch_size=32, X_val=X_test, Y_val=y_test)

