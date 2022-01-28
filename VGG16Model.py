from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

import tensorflow as tf
from tensorflow.keras import layers, models

class VGG16Model(Model):

    def __init__(self, name):
        super().__init__(name)

    def init_name(self):
        self.name = 'VGG16'

    def construct_model(self):

        model = models.Sequential()
        model.add(tf.keras.layers.Lambda( 
            lambda image: tf.image.resize( 
                image, 
                (224, 224), 
                # method = tf.image.ResizeMethod.BICUBIC,
                preserve_aspect_ratio = True
            )
        ))

        model.add(layers.Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
        model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
        model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(units = 4096, activation = 'relu'))
        model.add(layers.Dense(units = 4096, activation = 'relu'))
        model.add(layers.Dense(units = DataProperties.n_classes, activation = 'softmax'))

        self.model = model
       


        
