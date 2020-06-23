import tensorflow as tf
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock

class Transformer(tf.keras.Model):

    def __init__(self, query_dimensions_list_encoder, value_dimensions_list_encoder,
                 query_dimensions_list_decoder, value_dimensions_list_decoder, num_heads,
                 num_neurons_feedforward, encoding_dim_encoder, encoding_dim_decoder,
                 dropout_rate, num_timesteps, num_classes, num_encoder_blocks,
                 num_decoder_blocks, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.query_dimensions_list_encoder = query_dimensions_list_encoder
        self.value_dimensions_list_encoder = value_dimensions_list_encoder
        self.query_dimensions_list_decoder = query_dimensions_list_decoder
        self.value_dimensions_list_decoder = value_dimensions_list_decoder
        self.num_heads = num_heads
        self.num_neurons_feedforward = num_neurons_feedforward
        self.encoding_dim_encoder = encoding_dim_encoder
        self.encoding_dim_decoder = encoding_dim_decoder
        self.dropout_rate = dropout_rate
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks


        encoder_input = tf.keras.Input(shape=[encoding_dim_encoder], dtype=tf.float32)
        decoder_input = tf.keras.Input(shape=[encoding_dim_decoder], dtype=tf.float32)

        z = encoder_input
        for i in range(num_encoder_blocks):
            z = EncoderBlock(
                query_dimensions_list=query_dimensions_list_encoder,
                value_dimensions_list=value_dimensions_list_encoder,
                num_heads=num_heads,
                num_neurons_feedforward=num_neurons_feedforward,
                encoding_dim=encoding_dim_encoder,
                dropout_rate=dropout_rate
            )(z)
        encoder_output = z

        z = decoder_input
        for i in range(num_decoder_blocks):
            z = DecoderBlock(
                query_dimensions_list=query_dimensions_list_decoder,
                value_dimensions_list=value_dimensions_list_decoder,
                num_heads=num_heads,
                num_neurons_feedforward=num_neurons_feedforward,
                encoding_dim=encoding_dim_decoder,
                dropout_rate=dropout_rate,
                num_timesteps=num_timesteps
            )([z, encoder_output])
        decoder_output = z

        class_scores = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            input_dim=encoding_dim_decoder)(
                                            decoder_output
                                            )


        self.transformer = tf.keras.models.Model(inputs=[encoder_input, decoder_input],
                                                outputs=[class_scores],
                                                name="Transformer")

    def call(self, input):
        return self.transformer(input)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'query_dimensions_list_encoder': self.query_dimensions_list_encoder,
                'value_dimensions_list_encoder': self.value_dimensions_list_encoder,
                'query_dimensions_list_decoder': self.query_dimensions_list_decoder,
                'value_dimensions_list_decoder': self.value_dimensions_list_decoder,
                'num_heads': self.num_heads,
                'num_neurons_feedforward': self.num_neurons_feedforward,
                'encoding_dim_encoder': self.encoding_dim_encoder,
                'encoding_dim_decoder': self.encoding_dim_decoder,
                'dropout_rate': self.dropout_rate,
                'num_timesteps': self.num_timesteps,
                'num_classes': self.num_classes,
                'num_encoder_blocks': self.num_encoder_blocks,
                'num_decoder_blocks': self.num_decoder_blocks}
