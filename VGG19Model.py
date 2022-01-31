import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model as tf_Model

from Model import Model
from PreprocessingParameters import PreprocessingParameters
from DataProperties import DataProperties


class VGG19Model(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        # )(inputs)

        # x = Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(resize)
        # x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
        # x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

        # x = Flatten()(x)
        # x = Dense(units = 4096, activation = 'relu')(x)
        # x = Dense(units = 4096, activation = 'relu')(x)
        # predictions = Dense(units = DataProperties.n_classes, activation = 'softmax')(x)

        # self.model = tf_Model(inputs = inputs, outputs = predictions)

        model = tf.keras.models.Sequential()

        model.add(layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (224, 224, 3), kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation = 'relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation = 'relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation = 'softmax'))

        self.model = model

