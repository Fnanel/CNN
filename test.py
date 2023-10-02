import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


Lable = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                )


training_set = train_datagen.flow_from_directory('C:/Users/dator/Desktop/archive (2)/images/images/train',
                                                                                            
                                                 target_size=(48, 48),
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 color_mode='grayscale')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('C:/Users/dator/Desktop/archive (2)/images/images/validation',
                                            target_size = (48, 48),
                                            batch_size = 64,
                                            class_mode = 'categorical',
                                            color_mode='grayscale')



cnn= tf.keras.models.Sequential()

tf.keras.backend.set_image_data_format('channels_last')

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation= 'relu', input_shape = [48,48,1]))


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation= 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation= 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation= 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) 

cnn.add(tf.keras.layers.Flatten())


cnn.add(tf.keras.layers.Dense(units=300, activation= 'relu' ))

cnn.add(tf.keras.layers.Dense(units=300, activation= 'relu' ))

cnn.add(tf.keras.layers.Dense(units=300, activation= 'relu' ))

cnn.add(tf.keras.layers.Dense(units=7, activation= 'softmax' ))


cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(x= training_set, validation_data = test_set, epochs=10)


cnn.save('facial_expression.h5')

