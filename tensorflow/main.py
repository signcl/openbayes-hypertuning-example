import sys
import os
import argparse
import openbayestool
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

parser = argparse.ArgumentParser(description='hypertuning')
parser.add_argument('--input', help='input')
parser.add_argument('--filters', help='filters')
parser.add_argument('--dropout', help='dropout')
parser.add_argument('--nn', help='nn')
parser.add_argument('--opt', help='opt')
args = parser.parse_args()
print(args.filters, args.dropout, args.nn, args.opt)

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 10

dp_rate = float(args.dropout)
dense_num = int(float(args.nn))
# number of convolutional filters to use
nb_filters = int(float(args.filters))
# input image dimensions
img_rows, img_cols = 28, 28
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(args.input, "mnist.npz"))


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dp_rate))

model.add(Flatten())
model.add(Dense(dense_num))
model.add(Activation('relu'))
model.add(Dropout(dp_rate))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer=args.opt,
        metrics=['accuracy'])

class OpenBayesMetricsCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        """Print Training Metrics"""
        if batch % 5000 == 0:
            # 如果在 tensorflow 2.0 必须使用 accuracy 而不是 acc
            # openbayestool.log_metric('acc', float(logs.get('accuracy')))
            openbayestool.log_metric('precision', float(logs.get('acc')))
            openbayestool.log_metric('loss', float(logs.get('loss')))


tb_callback = keras.callbacks.TensorBoard(log_dir='./tf_dir')
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
        verbose=1, validation_data=(X_test, Y_test), callbacks=[tb_callback, OpenBayesMetricsCallback()])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(history.history['val_acc'])
print(history.history['val_acc'][-1])
