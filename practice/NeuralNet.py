from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import time

import tensorflow as tf
tf.enable_eager_execution()
if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation = 'sigmoid')

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        rnn(rnn_units, return_sequences = True, recurrent_initializer='glorot_uniform', stateful = True),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


SEQ_LEN = 100
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EMBED_DIM = 256
RNN_UNITS = 1024
EPOCHS = 3

path_to_file = 'shakes.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8', errors="ignore")

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(SEQ_LEN+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

examples_per_epoch = len(text)//SEQ_LEN
steps_per_epoch = examples_per_epoch//BATCH_SIZE

model = build_model(len(vocab), EMBED_DIM, RNN_UNITS, BATCH_SIZE)
model.compile(optimizer = tf.train.AdamOptimizer(), loss = loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix, save_weights_only = True)

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
model = build_model(len(vocab), EMBED_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

print(generate_text(model, start_string=u"So "))