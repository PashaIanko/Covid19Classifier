from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential


class AlexNetModel(Model):

    def __init__(self, name):
        super().__init__(name)

    def init_name(self):
        self.name = 'AlexNet'

    def construct_model(self):

        model = models.Sequential()
        model.add(tf.keras.layers.Lambda( 
            lambda image: tf.image.resize( 
                image, 
                (224, 224), 
                method = tf.image.ResizeMethod.BICUBIC,
                align_corners = True, # possibly important
                preserve_aspect_ratio = True
            )
        ))
        model.add(layers.Conv2D(96, 11, strides=4, padding='same', input_shape = (224, 224, 3)))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))

        model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))

        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(DataProperties.n_classes, activation='softmax'))

        self.model = model
       


        
