from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model as tf_Model


def conv2d_bn(X_input, filters, kernel_size, strides, padding='same', activation=None,
              name=None):
    
    # defining name basis
    conv_name_base = 'conv_'
    bn_name_base = 'bn_'

    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
               padding = padding, name = conv_name_base + name, 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = bn_name_base + name)(X)
    if activation is not None:
        X = Activation(activation)(X)
    return X

# FUNCTION: stem_block

def stem_block(X_input):

    # First conv 
    X = conv2d_bn(X_input, filters = 32, kernel_size = (3, 3), strides = (2, 2), 
                  padding = 'valid', activation='relu', name = 'stem_1th')
    
    # Second conv
    X = conv2d_bn(X, filters = 32, kernel_size = (3, 3), strides = (1, 1), 
                  padding = 'valid', activation='relu', name = 'stem_2nd')

    # Third conv
    X = conv2d_bn(X, filters = 64, kernel_size = (3, 3), strides = (1, 1), 
                  padding = 'same', activation='relu', name =  'stem_3rd')

    # First branch: max pooling
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                           padding = 'valid', name = 'stem_1stbranch_1')(X)

    # Second branch: conv
    branch2 = conv2d_bn(X, filters = 96, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'stem_1stbranch_2')

    # Concatenate (1) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)

    # First branch: 2 convs
    branch1 = conv2d_bn(X, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_1_1') 
    branch1 = conv2d_bn(branch1, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'valid', activation='relu', 
                        name = 'stem_2ndbranch_1_2') 
    
    # Second branch: 4 convs
    branch2 = conv2d_bn(X, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_1') 
    branch2 = conv2d_bn(branch2, filters = 64, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_2') 
    branch2 = conv2d_bn(branch2, filters = 64, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_3') 
    branch2 = conv2d_bn(branch2, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'valid', activation='relu', 
                        name = 'stem_2ndbranch_2_4') 

    # Concatenate (2) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)

    # First branch: conv
    branch1 = conv2d_bn(X, filters = 192, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'stem_3rdbranch_1')

    # Second branch: max pooling
    branch2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                           padding = 'valid', name = 'stem_3rdbranch_2')(X)

    # Concatenate (3) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)

    return X


def inception_a_block(X_input, base_name):
    """
    Implementation of the Inception-A block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ia_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 96, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 96, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_2')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_3')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    
    return X

def inception_b_block(X_input, base_name):

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ib_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 128, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 224, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 192, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_3')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_4')
    branch4 = conv2d_bn(branch4, filters = 256, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_5')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    
    return X

def inception_c_block(X_input, base_name):

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ic_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_1')
    branch3_1 = conv2d_bn(branch3, filters = 256, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_2')
    branch3_2 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 448, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 512, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_3')
    branch4_1 = conv2d_bn(branch4, filters = 256, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_4')
    branch4_2 = conv2d_bn(branch4, filters = 256, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_5')

    # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3_1, branch3_2, branch4_1, 
                          branch4_2], axis=3)
    
    return X

def reduction_a_block(X_input):

    # Branch 1
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = 'valid', name = 'ra_branch_1_1')(X_input)
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 384, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'ra_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 224, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'ra_branch_3_3')

    # Concatenate branch1, branch2 and branch3 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3], axis=3)
    
    return X

def reduction_b_block(X_input):

    # Branch 1
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = 'valid', name = 'rb_branch_1_1')(X_input)
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_2_1')
    branch2 = conv2d_bn(branch2, filters = 192, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_2_2')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 320, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_3')
    branch3 = conv2d_bn(branch3, filters = 320, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_3_4')

    # Concatenate branch1, branch2 and branch3 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3], axis=3)
    
    return X

from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

class InceptionModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_name(self):
        self.name = 'Inception'

    def construct_model(self):
        X_input = Input(
            shape = PreprocessingParameters.target_shape + \
                   PreprocessingParameters.n_color_channels
        )

        X = stem_block(X_input)

        # Four Inception A blocks
        X = inception_a_block(X, 'a1')
        X = inception_a_block(X, 'a2')
        X = inception_a_block(X, 'a3')
        X = inception_a_block(X, 'a4')

        # Reduction A block
        X = reduction_a_block(X)

        # Seven Inception B blocks
        X = inception_b_block(X, 'b1')
        X = inception_b_block(X, 'b2')
        X = inception_b_block(X, 'b3')
        X = inception_b_block(X, 'b4')
        X = inception_b_block(X, 'b5')
        X = inception_b_block(X, 'b6')
        X = inception_b_block(X, 'b7')

        # Reduction B block
        X = reduction_b_block(X)

        # Three Inception C blocks
        X = inception_c_block(X, 'c1')
        X = inception_c_block(X, 'c2')
        X = inception_c_block(X, 'c3')

        # AVGPOOL (1 line). Use "X = AveragePooling2D(...)(X)"
        kernel_pooling = X.get_shape()[1:3]
        X = AveragePooling2D(kernel_pooling, name='avg_pool')(X)
        X = Flatten()(X)

        # Dropout
        X = Dropout(rate = 0.2)(X)

        # Output layer
        X = Dense(
            DataProperties.n_classes, activation='sigmoid', name='fc')(X)
        
        ### END CODE HERE ###
        
        # Create model
        self.model = tf_Model(inputs = X_input, outputs = X, name='Inceptionv4')

    def compile_model(self):
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = self.metrics
        )
