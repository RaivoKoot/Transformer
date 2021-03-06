import tensorflow as tf
from MultiheadAttentionBlock import MultiheadAttentionBlock

class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, query_dimensions_list, value_dimensions_list, num_heads,
                 num_neurons_feedforward, encoding_dim, dropout_rate, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.query_dimensions_list = query_dimensions_list
        self.value_dimensions_list = value_dimensions_list
        self.num_heads = num_heads
        self.num_neurons_feedforward = num_neurons_feedforward
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate

        multihead_attention_block = MultiheadAttentionBlock(
            query_dimensions_list=query_dimensions_list,
            value_dimensions_list=value_dimensions_list,
            num_heads=num_heads,
            encoding_dim=encoding_dim
        )

        # Define the Feed Forward Network Block of the Encoder
        feedforward_block = tf.keras.models.Sequential()
        feedforward_block.add(tf.keras.layers.Dense(num_neurons_feedforward,
                                                    activation='relu',
                                                    input_dim=encoding_dim))
        feedforward_block.add(tf.keras.layers.Dense(encoding_dim))
        feedforward_block.add(tf.keras.layers.Dropout(dropout_rate))
        feedforward_block = feedforward_block


        # Use Keras functional API to piece together entire encoder Model
        input = tf.keras.Input(shape=[encoding_dim],
                               dtype=tf.float32)

        contextualized_input = multihead_attention_block([input, input, input])
        contextualized_input = tf.keras.layers.Dropout(dropout_rate)(contextualized_input)

        residual_output = tf.keras.layers.Add()([contextualized_input, input])
        feedforward_input = tf.keras.layers.LayerNormalization()(residual_output)

        feedforward_output = feedforward_block(feedforward_input)
        feedforward_output = tf.keras.layers.Dropout(dropout_rate)(feedforward_output)

        residual_output = tf.keras.layers.Add()([feedforward_input, feedforward_output])
        encoder_output = tf.keras.layers.LayerNormalization()(residual_output)

        self.encoder = tf.keras.models.Model(inputs=[input],
                                             outputs=[encoder_output],
                                             name="EncoderBlock")

    def call(self, input_batch):
        return self.encoder(input_batch)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'query_dimensions_list': self.query_dimensions_list,
                'value_dimensions_list': self.value_dimensions_list,
                'num_heads': self.num_heads,
                'num_neurons_feedforward': self.num_neurons_feedforward,
                'encoding_dim': self.encoding_dim,
                'dropout_rate': self.dropout_rate}
