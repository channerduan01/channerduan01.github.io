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

# %%
# define the dependency
from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# Requires at least tf-version: 1.11.0
print("tf-version:", tf.__version__)


# %%
# load the data
# choice the dataset
from enum import Enum
class DataSetChoice(Enum):
    MINIST = 1
    FASION_MINIST = 2
    CIFAR10 = 3

DATASET_CHOICE = DataSetChoice.FASION_MINIST

if DATASET_CHOICE == DataSetChoice.MINIST:
    raw_dataset = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = raw_dataset.load_data()
    class_names = ['%s' %x for x in range(10)] 
elif DATASET_CHOICE == DataSetChoice.FASION_MINIST:
    raw_dataset = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = raw_dataset.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
elif DATASET_CHOICE == DataSetChoice.CIFAR10:
    raw_dataset = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = raw_dataset.load_data()
    class_names = ['airplane','automobile','bird','car','deer','dog','frog','horse','ship','truck']
    # RGB to gray
    train_images = train_images[:,:,:,0]*0.299+train_images[:,:,:,1]*0.587+train_images[:,:,:,2]*0.114
    test_images = test_images[:,:,:,0]*0.299+test_images[:,:,:,1]*0.587+test_images[:,:,:,2]*0.114
else:
    raise Exception("bad dataset")
# normalization
train_images = (train_images - 255.0/2) / 255.0
test_images = (test_images - 255.0/2) / 255.0
# just show the shapes
print("train_images.shape:", train_images.shape)
print("train_labels.shape:", train_labels.shape)
print("test_images.shape:", test_images.shape)
print("test_labels.shape:", test_labels.shape)


# %%
# Some visualization
IDX_IMAGE = 40
plt.figure(figsize=(2,2))
plt.axis("off")
plt.title('type: %s' %class_names[train_labels[IDX_IMAGE]])
plt.imshow(train_images[IDX_IMAGE], cmap=plt.cm.binary)
plt.show()


# %%
# Some more visualization
IDX_START = 150

plt.figure(figsize=(10,10))
for i in range(IDX_START,IDX_START+25):
    plt.subplot(5,5,i-IDX_START+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# %%
# Define a simple NN model
USING_SIMPLE_MODEL = False

shape_of_image = train_images.shape[1:]
if USING_SIMPLE_MODEL:
    # basic model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=shape_of_image),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
else:
    # a better model
    from tensorflow.keras import backend as K
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: K.reshape(x,[-1] + list(shape_of_image) + [1]), name='expand_channel', dtype='float32', input_shape=(28,28)))
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


# %%
# train the model
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.1, batch_size=32)

# draw the loss curve
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)
plt.clf()   # clear figure
plt.plot(epochs, history.history['loss'], 'ro', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

    
# %%
# evaluate manually 
IDX_TEST_IMAGE = 40

predict_dist = model.predict(np.expand_dims(test_images[IDX_TEST_IMAGE],0))
predicted_label = np.argmax(predict_dist)
true_label = test_labels[IDX_TEST_IMAGE]
# show the case
plt.figure(figsize=(2,2))
plt.axis("off")
plt.title('predict_type: %s, actual: %s' %(class_names[predicted_label], class_names[true_label]))
plt.imshow(test_images[IDX_TEST_IMAGE], cmap=plt.cm.binary)
plt.show()
# show the predict distribution
thisplot = plt.bar(range(10), predict_dist[0], color="#777777")
plt.ylim([0, 1])
plt.xticks([])
thisplot[predicted_label].set_color('red')
thisplot[test_labels[IDX_TEST_IMAGE]].set_color('blue')
_ = plt.xticks(range(10), class_names, rotation=45)


# %%
# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

    

    
    
    
    
    
    
    
    
