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


#%% define the dependency

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)


#%% load the data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

#%%


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#%%




#%%




#%%




#%%




#%%




