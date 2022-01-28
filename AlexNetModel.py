from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model as tf_Model


class AlexNetModel(Model):

    def __init__(self, name):
        super().__init__(name)

    def init_name(self):
        self.name = 'AlexNet'

    def construct_model(self):

        inputs = layers.Input(
            shape = PreprocessingParameters.target_shape + \
                   PreprocessingParameters.n_color_channels
        )

        resize = tf.keras.layers.Lambda(
            lambda image: tf.image.resize(
                image,
                (256, 256),
                preserve_aspect_ratio = True
            )
        )(inputs)

        x = layers.Conv2D(96, 11, strides=4, padding='same', input_shape = (256, 256, 3))(resize)
        x = layers.Lambda(tf.nn.local_response_normalization)(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        x = layers.Conv2D(256, 5, strides=4, padding='same')(x)
        x = layers.Lambda(tf.nn.local_response_normalization)(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2)(x)

        x = layers.Conv2D(384, 3, strides=4, padding='same')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(384, 3, strides=4, padding='same')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(256, 3, strides=4, padding='same')(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(DataProperties.n_classes, activation='softmax')(x)

        self.model = tf_Model(inputs = inputs, outputs = predictions)
       


        
