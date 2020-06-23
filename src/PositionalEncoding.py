import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

class PositionalEncoding(tf.keras.layers.Layer):
    """
    This code is mostly taken from Tensorflow Documentation "Transformer
    model for language understanding".
    """

    def __init__(self, steps, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        angle_rads = get_angles(np.arange(steps)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        #pos_encoding = angle_rads[np.newaxis, ...]
        self.pos_encoding = tf.cast(angle_rads, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding
