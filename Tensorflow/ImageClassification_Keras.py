# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:27:51 2020

@author: Guru Prasad Muppana

We will use Keras for image classification .

References: https://www.geeksforgeeks.org/python-image-classification-using-keras/


Data Set is at :
    https://drive.google.com/open?id=1dbcWabr3Xrr4JvuG0VxTiweGzHn-YYvW
    

"""


import numpy as np


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D # CNN specific layers
from keras.layers import Dense, Dropout, Flatten, Activation

from keras import backend as K

# images size : 224*224

train_data_dir = 'v_data/train'
test_data_dir = 'v_data/test'

nb_train_samples =400
nb_test_samples = 100

epochs = 10
batch_size = 16

# Image specifications:
img_width = 224
img_height = 224

# checking the number of channel in images
if K.image_data_format() == "channels_first" :
    input_shape = (3, img_width,img_height)
else:
    input_shape = (img_width,img_height,3) # this is a typical format.
print(input_shape)


model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (224,224,3))) 
# use : input_shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (2,2))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (2,2))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'rmsprop',
        metrics=["accuracy"]
        )


# Read the images from the folders:
train_datagen = ImageDataGenerator (
        rescale = 1. /225,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

test_datagen = ImageDataGenerator (
        rescale = 1./ 255
        )

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width,img_height),
        batch_size = batch_size,
        class_mode= 'binary'
        )

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width,img_height),
        batch_size = batch_size,
        class_mode= 'binary'
        )

# training is happening here.
model.fit_generator(
      train_generator,
      steps_per_epoch = nb_train_samples // batch_size,
      epochs = epochs,
      validation_data = test_generator,
      validation_steps = nb_test_samples // batch_size
      )

model.save_weights('model_saved.h5') # // where to place this?


