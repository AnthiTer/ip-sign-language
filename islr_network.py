from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout, Softmax
from tensorflow.keras.models import Sequential
import numpy as np

# actions = np.array(['goodbye', 'hi', 'you', 'me', 'thankyou', 'goodmorning'])

actions = np.array(['come', 'good', 'happy', 'home', 'iloveyou', 'sorry'])

# actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

# actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'sorry', 'thankyou'])

def islr():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(30, 258)))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(LSTM(128, return_sequences=True))
    model.add(LeakyReLU())
    model.add(LSTM(64, return_sequences=False))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(actions.shape[0]))
    model.add(Softmax())

    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model