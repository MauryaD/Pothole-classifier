# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:28:18 2020

@author: Deepak Maurya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:36:51 2020

@author: Deepak Maurya
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#part 1 CNN formation

#initialisation of CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening
classifier.add(Flatten())

#connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))     #if more than 2 output then use softmax as activation func.

#compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] ) #if more than 2 output then use cross entropy as loss func.


#part 2 Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale=  1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set, 
                        steps_per_epoch=948,
                        epochs=8,
                        validation_data=test_set,
                        validation_steps=526)