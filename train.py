import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Softmax, Dropout, GRU
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, History, ModelCheckpoint
import tensorflow as tf
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

# actions = np.array(['goodbye', 'hi', 'you', 'me', 'thankyou', 'goodmorning'])
# actions = np.array(['come', 'good', 'happy', 'home', 'iloveyou', 'sorry'])
actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

sequences = np.load('seq_labels/sequences_pose_all.npy')
labels = np.load('seq_labels/labels_pose_all.npy')
 
np.random.seed(24)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

scores = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1'])

def train_eval(a):
    file_name = f'all_gru_{a}'

    xtrain, xrem, ytrain, yrem = train_test_split(X, y, test_size=0.3)

    xval, xtest, yval, ytest = train_test_split(xrem, yrem, test_size=0.5)

    # np.random.seed(42)
    tf.random.set_seed(42)
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(f'islr_{file_name}.h5', monitor='val_loss', save_best_only=True)
    history = History()
    callback = EarlyStopping(monitor='val_loss', patience=7, start_from_epoch = 80)

    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=(30, 258)))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(GRU(128, return_sequences=True))
    model.add(LeakyReLU())
    model.add(GRU(64, return_sequences=False))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(actions.shape[0]))
    model.add(Softmax())
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()

    history = model.fit(xtrain, ytrain, epochs=100, callbacks=[tb_callback, callback, history, checkpoint_callback], batch_size=32, validation_data=(xval, yval))

    model.load_weights(f'islr_{file_name}.h5')
    
    train_loss, train_acc = model.evaluate(xtrain, ytrain)
    print('Train accuracy: ', train_acc)
    print('Train loss: ', train_loss)

    val_loss, val_acc = model.evaluate(xval, yval)
    print('Val accuracy: ', val_acc)
    print('Val loss: ', val_loss)

    test_loss, test_acc = model.evaluate(xtest, ytest)
    print('Test accuracy: ', test_acc)
    print('Test loss: ', test_loss)

    yhat = model.predict(xtest)

    ytrue = np.argmax(ytest, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    cm = confusion_matrix(ytrue,yhat)

    cm_df = pd.DataFrame(cm,
                     index = actions, 
                     columns = actions)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(f'graphs_eval/conf_matrix_{file_name}.png')

    pd.DataFrame(history.history).to_csv(f'graphs_eval/history_{file_name}.csv', index=False)

    acc = accuracy_score(ytrue, yhat)
    rec = recall_score(ytrue, yhat, average = 'weighted')
    pre = precision_score(ytrue, yhat, average = 'weighted')
    f1 = f1_score(ytrue, yhat, average = 'weighted')
    f1_each = f1_score(ytrue, yhat, average = None)

    scores.loc[a] = [acc, rec, pre, f1]

    return f1_each

num_runs = 10
results = [train_eval(a+10) for a in range(num_runs)]
pd.options.display.float_format = '{:.4f}'.format
pd.DataFrame(results, columns=actions).to_csv('error_bars_gru_1.csv', index=False)
scores.to_csv('scores_gru_1.csv', index=False)