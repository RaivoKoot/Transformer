import tensorflow as tf
from MultiheadAttentionBlock import MultiheadAttentionBlock

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, query_dimensions_list, value_dimensions_list, num_heads, encoding_dim=300, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        # Compute multihead attention
        # apply dropout
        # residual add and normalize
        # apply 2 layer feed-forward network
        # apply dropout
        # residual add and normalize

    def call(self, inputs_list):
        pass
