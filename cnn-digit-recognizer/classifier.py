#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digit recognizer using CNN
training set size: 60000
testing set size: 10000
Each image has 28 x 28 resolution.
"""
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
from keras.models import Sequential # initinalize the neural network
from keras.layers import Conv2D # used to add convolutional layer
from keras.layers import MaxPooling2D # used to add pooling layer
from keras.layers import Flatten # used for flatterning
from keras.layers import Dense # used to add fully connected layers in a classic ANN
from keras.layers import Dropout# add dropout regularization to prevent overfitting

# Part 1 - building a CNN
# Initialising the CNN
classifier = Sequential()

# Add 1st convolutional + pooling layer
# Convolution
classifier.add(Conv2D(filters=32, 
                      kernel_size= (3, 3), 
                      strides=1,
                      input_shape=(28, 28, 1), 
                      data_format='channels_last', 
                      activation='relu')) 

classifier.add(Conv2D(filters=32,
                       kernel_size=(3, 3),
                       strides=1,
                       data_format='channels_last', 
                       activation='relu')) 

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Dropout(0.25))
# Add 2nd convolutional + pooling layer
# Convolution
classifier.add(Conv2D(filters=64, 
                      kernel_size=(3, 3), 
                      strides=1, 
                      data_format='channels_last', 
                      activation = 'relu')) 

classifier.add(Conv2D(filters=64, 
                       kernel_size=(3, 3),
                       strides=1,
                       data_format='channels_last', 
                       activation = 'relu')) 

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Dropout(0.25))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 512, activation = 'relu')) # fully connected hidden layer
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 10, activation = 'softmax')) # output layer
#classifier.summary()

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
batch_size = 64

from keras.preprocessing.image import ImageDataGenerator # Data Augmentation: to reduce overfitting

train_datagen = ImageDataGenerator(rescale = 1./255,# all pixel values will be in the range of [0, 1]
                                   shear_range = 0.2,
                                   zoom_range = 0.2, # Randomly zoom image 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('mnist_png/training_set',
                                                 target_size = (28, 28),
                                                 color_mode="grayscale",
                                                 batch_size = batch_size,
                                                 class_mode="categorical")

test_set = test_datagen.flow_from_directory('mnist_png/test_set',
                                            target_size = (28, 28),
                                            color_mode="grayscale",
                                            batch_size = batch_size,
                                            class_mode="categorical")
 

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 60000//batch_size,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 10000//batch_size)


 # Plot train and validation curves
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()



