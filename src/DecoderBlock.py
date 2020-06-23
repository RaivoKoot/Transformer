import tensorflow as tf
from MultiheadAttentionBlock import MultiheadAttentionBlock

class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, query_dimensions_list, value_dimensions_list, num_heads,
                 num_neurons_feedforward, encoding_dim, dropout_rate, num_timesteps, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.query_dimensions_list = query_dimensions_list
        self.value_dimensions_list = value_dimensions_list
        self.num_heads = num_heads
        self.num_neurons_feedforward = num_neurons_feedforward
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.num_timesteps = num_timesteps

        masked_multihead_attention_block = MultiheadAttentionBlock(
            query_dimensions_list=query_dimensions_list,
            value_dimensions_list=value_dimensions_list,
            num_heads=num_heads,
            encoding_dim=encoding_dim,
            use_masking=True,
            num_timesteps=num_timesteps
        )

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


        # Use Keras functional API to piece together entire Decoder Model
        input = tf.keras.Input(shape=[encoding_dim],
                               dtype=tf.float32)
        encoder_output = tf.keras.Input(shape=[encoding_dim],
                               dtype=tf.float32)

        # First Attention Block
        contextualized_input = masked_multihead_attention_block([input, input, input])
        contextualized_input = tf.keras.layers.Dropout(dropout_rate)(contextualized_input)

        residual_output = tf.keras.layers.Add()([contextualized_input, input])
        attention2_queries = tf.keras.layers.LayerNormalization()(residual_output)


        # Second Attention Block where encoder ouput and decoder input is mixed
        attention2_input_contextualized = \
                multihead_attention_block([attention2_queries, encoder_output, encoder_output])
        attention2_input_contextualized = \
                tf.keras.layers.Dropout(dropout_rate)(attention2_input_contextualized)

        residual_output = tf.keras.layers.Add()([attention2_input_contextualized, attention2_queries])
        feedforward_input = tf.keras.layers.LayerNormalization()(residual_output)


        # Feed Forward Block
        feedforward_output = feedforward_block(feedforward_input)
        feedforward_output = tf.keras.layers.Dropout(dropout_rate)(feedforward_output)

        residual_output = tf.keras.layers.Add()([feedforward_input, feedforward_output])
        decoder_output = tf.keras.layers.LayerNormalization()(residual_output)

        self.decoder = tf.keras.models.Model(inputs=[input, encoder_output],
                                             outputs=[decoder_output],
                                             name="DecoderBlock")

    def call(self, input_batch):
        return self.decoder(input_batch)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'query_dimensions_list': self.query_dimensions_list,
                'value_dimensions_list': self.value_dimensions_list,
                'num_heads': self.num_heads,
                'num_neurons_feedforward': self.num_neurons_feedforward,
                'encoding_dim': self.encoding_dim,
                'dropout_rate': self.dropout_rate,
                'num_timesteps': self.num_timesteps}
