import tensorflow as tf
from AttentionBlock import AttentionBlock

class MultiheadAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, query_dimensions_list, value_dimensions_list, num_heads,
                 use_masking=False, num_timesteps=-1, encoding_dim=300, **kwargs):
        super(MultiheadAttentionBlock, self).__init__(**kwargs)

        self.query_dimensions_list = query_dimensions_list
        self.value_dimensions_list = value_dimensions_list
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.num_timesteps = num_timesteps
        self.encoding_dim = encoding_dim

        # Set up the multihead attention block as a keras Model
        # using the Functional API. The batch dimension will be the sequence length.
        query_input = tf.keras.Input(shape=[encoding_dim],
                                           dtype=tf.float32,
                                           name='pre_query_matrix')
        key_input = tf.keras.Input(shape=[encoding_dim],
                                           dtype=tf.float32,
                                           name='pre_key_matrix')
        value_input = tf.keras.Input(shape=[encoding_dim],
                                           dtype=tf.float32,
                                           name='pre_value_matrix')

        # Initialize the attention heads
        attention_outputs = []
        for i in range(num_heads):
            query_dimensions = query_dimensions_list[i]
            value_dimensions = value_dimensions_list[i]
            attention_output = AttentionBlock(query_dimensions,
                                             value_dimensions,
                                             use_masking=use_masking,
                                             num_timesteps=num_timesteps,
                                             encoding_dim=encoding_dim)([query_input, key_input, value_input])

            attention_outputs.append(attention_output)

        attentions_concatenated = tf.keras.layers.Concatenate(axis=1)(attention_outputs)

        output = tf.keras.layers.Dense(encoding_dim,
                                       input_dim=sum(value_dimensions_list)
                                       )(attentions_concatenated)

        self.multihead_computation = tf.keras.models.Model(inputs=[query_input, key_input, value_input],
                                                           outputs=output,
                                                           name='multihead_block')

    def call(self, inputs_list):
        return self.multihead_computation(inputs_list)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'query_dimensions_list': self.query_dimensions_list,
                'value_dimensions_list': self.value_dimensions_list,
                'num_heads': self.num_heads,
                'use_masking': self.use_masking,
                'num_timesteps': self.num_timesteps,
                'encoding_dim': self.encoding_dim}
