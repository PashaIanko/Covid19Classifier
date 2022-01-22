import tensorflow as tf

def normalize(image):
    return image / 255.
    # return tf.cast(image, tf.float32) / 255.