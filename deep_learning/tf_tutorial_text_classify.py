#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:11:04 2019

Tensorflow keras tutorial - Text sentiment classification

Reference:
1. tf official tutorial
https://www.tensorflow.org/tutorials/keras/basic_text_classification


@author: channerduan
"""


# %%
# define the dependency
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)


# %%
# load the data
imdb = keras.datasets.imdb
# The data (only with top 10000 frequency words)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
print("Training entries: {}, labels: {}, size of word: {}".format(len(train_data), len(train_labels), len(word_index)))

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# %%
# Get some sentence
idx_sentence = 0
# the sentence has already encoded
print("raw encode:", train_data[idx_sentence])
print("decode:", decode_review(train_data[idx_sentence]))
# and the length of sentences are not equal
print("length of sentences:", len(train_data[0]), len(train_data[1]))


# %%
# Pad the sentence
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# %%
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# %%
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# %%
results = model.evaluate(test_data, test_labels)
print(results)


# %%
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'ro', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# %%
plt.clf()   # clear figure

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


#%%







