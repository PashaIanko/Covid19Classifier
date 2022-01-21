from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties
from tensorflow.keras.layers import ReLU as ReLU
from tensorflow.keras.layers import MaxPool2D as MaxPool2D
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Flatten as Flatten
from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.layers import Input as Input
import tensorflow as tf


def conv_2d_pooling_layers(n_filters):
    return [
            Conv2D(
                filters = n_filters,
                kernel_size = (3, 3),
                padding = 'same',
                activation = 'elu'
            ),
            MaxPool2D()
    ]

class CNNModel(Model):

    def __init__(self):
        super().__init__()

    def init_name(self):
        self.name = 'CNN'

    def construct_model(self):
        input_layer = [
               Input(shape = PreprocessingParameters.target_shape + \
                   PreprocessingParameters.n_color_channels)
        ]

        core_layers = conv_2d_pooling_layers(16) + \
            conv_2d_pooling_layers(32) + \
            conv_2d_pooling_layers(64) + \
            conv_2d_pooling_layers(256)

        dense_layers = [
                Flatten(),
                Dense(128, activation = 'elu'),
                Dense(DataProperties.n_classes, activation = 'softmax')
        ]

        cnn_model = tf.keras.Sequential(
            input_layer + \
            core_layers + \
            dense_layers
        )
        
        self.model = cnn_model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )


        
