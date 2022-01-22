# parent class
from Model import Model

# parameters
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

# layers
from tensorflow.keras.layers import Input as Input
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import BatchNormalization as BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D as MaxPool2D
from tensorflow.keras.layers import Flatten as Flatten
from tensorflow.keras.layers import Dense as Dense

import tensorflow as tf


class BNModel(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_name(self):
        self.name = 'BN CNN'

    def bn_conv_pool_layer(self, filters, kernel_size, strides):
        return [
            Conv2D(
                filters, 
                kernel_size = kernel_size, 
                strides = strides, 
                padding = 'same', 
                activation = None
            ),
            
            BatchNormalization(axis = -1),
            
            Activation('relu'),
            
            MaxPool2D(
                pool_size = (2, 2),
                strides = (2, 2),
                padding = 'same'
            )
        ]


    def construct_model(self):
        input_layer = [
                Input(
                   shape = PreprocessingParameters.target_shape + \
                   PreprocessingParameters.n_color_channels
                )
        ]

        core_layers = \
            self.bn_conv_pool_layer(16, (3, 3), (1, 1)) + \
            self.bn_conv_pool_layer(32, (3, 3), (1, 1)) + \
            self.bn_conv_pool_layer(64, (3, 3), (1, 1)) + \
            self.bn_conv_pool_layer(256, (3, 3), (1, 1))

        dense_layers = [
                Flatten(),
                Dense(128, activation = 'elu'),
                Dense(DataProperties.n_classes, activation = 'softmax')
        ]

        bn_model = tf.keras.Sequential(
            input_layer + \
            core_layers + \
            dense_layers
        )
        
        self.model = bn_model

    def compile_model(self):
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = self.metrics
        )


        
