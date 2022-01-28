from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties
from tensorflow.keras.layers import ReLU as ReLU
from tensorflow.keras.layers import MaxPool2D as MaxPool2D
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Flatten as Flatten
from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.layers import Input as Input
from tensorflow.keras.regularizers import L1L2
import tensorflow as tf


def conv_2d_pooling_layers(n_filters):
    return [
            Conv2D(
                filters = n_filters,
                kernel_size = (3, 3),
                padding = 'same',
                activation = 'relu',
                kernel_regularizer = L1L2(l1=0.01, l2=0.01)
            ),
            MaxPool2D()
    ]

class CNNModel(Model):

    def __init__(self, name):
        super().__init__(name)

    def init_name(self):
        self.name = 'CNN'

    def construct_model(self):
        input_layer = [
               Input(shape = PreprocessingParameters.target_shape + \
                   PreprocessingParameters.n_color_channels)
        ]

        core_layers = \
            conv_2d_pooling_layers(16) + \
            conv_2d_pooling_layers(32) + \
            conv_2d_pooling_layers(64) + \
            conv_2d_pooling_layers(256)

        dense_layers = [
                Flatten(),
                Dense(128, activation = 'relu', kernel_regularizer = L1L2(l1 = 0.0, l2 = 0.01)),
                Dense(DataProperties.n_classes, activation = 'softmax')
        ]

        cnn_model = tf.keras.Sequential(
            input_layer + \
            core_layers + \
            dense_layers
        )
        
        self.model = cnn_model
       


        
