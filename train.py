import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os

sequences = np.load('sequences_pose.npy')
labels = np.load('labels_pose.npy')
actions = np.array(['goodbye', 'hi', 'you', 'me', 'thankyou', 'goodmorning'])

X = np.array(sequences)
# labels_tensor = (labels, dtype=torch.int64).clone().detach()
# y = torch.nn.functional.one_hot(torch.tensor(labels_tensor)).to(torch.float32)
# labels_tensor = torch.tensor(labels, dtype=torch.int64)
# labels_tensor_copy = labels_tensor.clone().detach()
# y = torch.nn.functional.one_hot(labels_tensor_copy).to(torch.float32)
y = to_categorical(labels).astype(int)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3)

X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

model.save('action.h5')

model.load_weights('action.h5')