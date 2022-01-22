from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model as tf_Model

from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties



def identity_block(X_input, f, filters, stage, block):

    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # First component of main path (3 lines)
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'valid', name = conv_name_base + '1th', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = bn_name_base + '1th')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), 
               padding = 'same', name = conv_name_base + '2nd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2nd')(X)
    X = Activation('relu')(X)

    # Third component of main path (2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'valid', name = conv_name_base + '3rd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '3rd')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (2 lines)
    X = Add()([X_input, X])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X_input, f, filters, stage, block, s = 2):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters

    ##### MAIN PATH ##### 
    # First component of main path (3 lines)
    X = Conv2D(F1, (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1st', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = bn_name_base + '1st')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (3 lines)
    X = Conv2D(F2, (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2nd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2nd')(X)
    X = Activation('relu')(X)

    # Third component of main path (2 lines)
    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '3rd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '3rd')(X)

    ##### SHORTCUT PATH #### (2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

class ResNetModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init__name(self):
        self.name = 'ResNet'

    def construct_model(self):
        X_input = Input(
            shape = PreprocessingParameters.target_shape + \
                PreprocessingParameters.n_color_channels
        )

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL
        X = AveragePooling2D(name='avg_pool')(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(
            DataProperties.n_classes, activation='sigmoid', name='fc', kernel_initializer = glorot_uniform(seed=0)
            )(X)

        # Create model
        self.model = tf_Model(inputs = X_input, outputs = X, name='ResNet50')
    
    def compile_model(self):
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = self.metrics
        )


