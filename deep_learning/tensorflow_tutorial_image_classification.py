#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:58:23 2019

Tensorflow keras tutorial - Image classification

Reference:
1. tf official tutorial
https://www.tensorflow.org/tutorials/keras/basic_classification
2. good blog for this example
https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
3. tf keras doc
https://www.tensorflow.org/api_docs/python/tf/keras
4. keras doc
https://keras.io/models/sequential/
5. a very very very good online course for this example 
http://cs231n.github.io/convolutional-networks/#layerpat


@author: channerduan
"""

#%% define the dependency
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Requires at least tf-version: 1.11.0
print("tf-version:", tf.__version__)


#%% load the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# normalization
train_images = (train_images - 255.0/2) / 255.0
test_images = (test_images - 255.0/2) / 255.0
# just show the shapes
print("train_images.shape:", train_images.shape)
print("train_labels.shape:", train_labels.shape)
print("test_images.shape:", test_images.shape)
print("test_labels.shape:", test_labels.shape)


#%% Some visualization
idx_image = 40
plt.figure(figsize=(2,2))
plt.axis("off")
plt.title('type: %s' %class_names[train_labels[idx_image]])
plt.imshow(train_images[idx_image], cmap=plt.cm.binary)
plt.show()


#%% Some more visualization
idx_start = 150
plt.figure(figsize=(10,10))
for i in range(idx_start,idx_start+25):
    plt.subplot(5,5,i-idx_start+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#%% Define a simple NN model
USING_SIMPLE_MODEL = True
if USING_SIMPLE_MODEL:
    # basic model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
else:
    # a better model
    from tensorflow.keras import backend as K
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: K.reshape(x,[-1,28,28,1]), name='expand_channel', dtype='float32', input_shape=(28,28)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


#%% train the model
history = model.fit(train_images, train_labels, epochs=5)
plt.plot(history.history['loss'])

    
#%% evaluate manually 
idx_test_image = 40
predict_dist = model.predict(np.expand_dims(test_images[idx_test_image],0))
predicted_label = np.argmax(predict_dist)
true_label = test_labels[idx_test_image]
# show the case
plt.figure(figsize=(2,2))
plt.axis("off")
plt.title('predict_type: %s, actual: %s' %(class_names[predicted_label], class_names[true_label]))
plt.imshow(test_images[idx_test_image], cmap=plt.cm.binary)
plt.show()
# show the predict distribution
thisplot = plt.bar(range(10), predict_dist[0], color="#777777")
plt.ylim([0, 1])
plt.xticks([])
thisplot[predicted_label].set_color('red')
thisplot[test_labels[idx_test_image]].set_color('blue')
_ = plt.xticks(range(10), class_names, rotation=45)


#%% evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

    
#%%

    
    
    
    
    
    
    
    
