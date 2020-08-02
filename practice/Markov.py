from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_index in enumerate(sequences):
        results[i, word_index] = 1.0
    return results

def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize = (16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Validation')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(), label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

VOCAB_SIZE = 10000

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = multi_hot_sequences(train_data, dimension = VOCAB_SIZE)
test_data = multi_hot_sequences(test_data, dimension = VOCAB_SIZE)

base_model = keras.Sequential([
    keras.layers.Dense(16, activation = tf.nn.relu, input_shape = (VOCAB_SIZE,)),
    keras.layers.Dense(16, activation = tf.nn.relu),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
base_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy', 'binary_crossentropy'])

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(VOCAB_SIZE,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(VOCAB_SIZE,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

base_history = base_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
l2_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
dpt_history = dpt_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

plot_history([('baseline', base_history),
              ('l2', l2_history),
              ('dpt', dpt_history)], key = 'acc')