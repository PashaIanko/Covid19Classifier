from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model as tf_Model


class VGG16Model(Model):

    def __init__(self, name):
        super().__init__(name)

    def init_name(self):
        self.name = 'VGG16'

    def construct_model(self):

        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer = 'he_normal'))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=3, activation="softmax"))

        self.model = model
        
       


        
