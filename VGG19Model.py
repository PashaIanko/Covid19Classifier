from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model as tf_Model

from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties


class VGG19Model(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.block1 = dict(
            filters = 64,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same'
        )

        self.block2 = dict(
            filters = 128,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same'
        )

        self.block3 = dict(
            filters = 256,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same'
        )

        self.block4 = dict(
            filters = 512,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same'
        )

        self.block5 = dict(
            filters = 512,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same'
        )

    def construct_model(self):
        X_input = Input(
            shape = PreprocessingParameters.target_shape + \
                PreprocessingParameters.n_color_channels
        )

        X = Conv2D(**self.block1)(X_input)
        X = Conv2D(**self.block1)(X)
        X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(X)

        X = Conv2D(**self.block2)(X)
        X = Conv2D(**self.block2)(X)
        X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(X)

        X = Conv2D(**self.block3)(X)
        X = Conv2D(**self.block3)(X)
        X = Conv2D(**self.block3)(X)
        X = Conv2D(**self.block3)(X)
        X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(X)

        X = Conv2D(**self.block4)(X)
        X = Conv2D(**self.block4)(X)
        X = Conv2D(**self.block4)(X)
        X = Conv2D(**self.block4)(X)
        X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(X)

        X = Conv2D(**self.block5)(X)
        X = Conv2D(**self.block5)(X)
        X = Conv2D(**self.block5)(X)
        X = Conv2D(**self.block5)(X)
        X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(X)

        X = Flatten()(X)
        X = Dense(units = 4096, activation = 'relu')(X)
        X = Dropout(0.3)(X)
        X = Dense(units = 4096, activation = 'relu')(X)
        X = Dropout(0.3)(X)
        X = Dense(DataProperties.n_classes, activation='sigmoid')(X)
        
        self.model = tf_Model(inputs = X_input, outputs = X, name = self.name)


