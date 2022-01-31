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

        # inputs = layers.Input(
        #     shape = PreprocessingParameters.target_shape + \
        #         PreprocessingParameters.n_color_channels
        # )

        # resize = tf.keras.layers.Lambda(
        #     lambda image: tf.image.resize(
        #         image,
        #         (224, 224),
        #         preserve_aspect_ratio = True
        #     )
        # )

        # resize = tf.keras.layers.Lambda(
        #     lambda image: tf.image.resize(
        #         image,
        #         (224, 224),
        #         preserve_aspect_ratio = True
        #     )
        # )

        # preprocess = tf.keras.layers.Lambda(
        #     lambda image: tf.keras.applications.vgg16.preprocess_input(image)
        # )

        # vgg = tf.keras.applications.vgg16.VGG16(
        #     include_top = True,
        #     weights = None,  # random initialization
        #     input_shape = None,  # must be (224, 224, 3)
        #     classes = DataProperties.n_classes,
        #     # classifier_activation = 'softmax'
        # )

        # self.model = tf.keras.Sequential([
        #     resize,
        #     preprocess,
        #     vgg
        # ])

        # self.model = vgg

        # self.model = tf.keras.Sequential([
        #     inputs, 
        #     resize,
        #     vgg
        # ])

        # inputs = layers.Input(
        #     shape = PreprocessingParameters.target_shape + \
        #         PreprocessingParameters.n_color_channels
        # )

        # resize = tf.keras.layers.Lambda(
        #     lambda image: tf.image.resize(
        #         image,
        #         (224, 224),
        #         preserve_aspect_ratio = True
        #     )
        # )(inputs)




        # inputs = layers.Input(
        #     shape = (224, 224, 3)
        # )

        # x = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(inputs)
        # x = layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
        # x = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

        # x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
        # x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

        # x = layers.Flatten()(x)
        # x = layers.Dense(units = 4096, activation = 'relu')(x)
        # x = layers.Dense(units = 4096, activation = 'relu')(x)
        # predictions = layers.Dense(units = DataProperties.n_classes, activation = 'softmax')(x)

        # self.model = tf_Model(inputs = inputs, outputs = predictions)

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
        
       


        
