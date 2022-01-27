import tensorflow as tf

def preprocess(image):
    return image / 255.
    # return tf.cast(image, tf.float32) / 255.