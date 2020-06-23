import tensorflow as tf

class AttentionBlock(tf.keras.layers.Layer):

    def __init__(self, query_dimensions, value_dimensions, encoding_dim,
                 use_masking=False, num_timesteps=-1, **kwargs):

        super(AttentionBlock, self).__init__(**kwargs)

        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.encoding_dim = encoding_dim
        self.use_masking = use_masking
        self.num_timesteps = num_timesteps

        self.Wq = self.add_weight(
            shape=(encoding_dim, query_dimensions), initializer='random_normal',
            trainable=True,
            name='Wqueries'
        )

        self.Wk = self.add_weight(
            shape=(encoding_dim, query_dimensions), initializer='random_normal',
            trainable=True,
            name='Wkeys'
        )

        self.Wv = self.add_weight(
            shape=(encoding_dim, value_dimensions), initializer='random_normal',
            trainable=True,
            name='Wvalues'
        )


        self.scaling_term = tf.math.sqrt(tf.constant(query_dimensions, dtype=tf.float32))

        if not use_masking:
            self.masking_matrix = tf.constant(0., dtype=tf.float32)
        elif num_timesteps == -1:
            raise ValueError('When setting use_masking to true, you must set num_timesteps')
        else:
            self.build_masking_matrix(num_timesteps)


    def build_masking_matrix(self, num_timesteps):
        diagonal_values = tf.fill((num_timesteps-1, num_timesteps-1), float('-inf'))
        diagonal_range = (1, num_timesteps-1) # Diagonal number 1 (one above main diagonal)
                                              # until last upper diagonal

        self.masking_matrix = tf.linalg.diag(
                                    diagonal_values, k=diagonal_range,
                                    num_rows=num_timesteps, num_cols=num_timesteps)

        # In case 1 == num_timesteps-1, tf.linalg.diag returns wron dimensions
        self.masking_matrix = tf.reshape(self.masking_matrix, (num_timesteps, num_timesteps))

    def call(self, inputs_list):
        """
        This computes the attention over a sequence. Each
        tensor in the inputs_list parameter is of shape
        (timesteps, encoding_dim). Does not handle batched
        input sequences.
        """
        pre_query_matrix = inputs_list[0]
        pre_key_matrix = inputs_list[1]
        pre_value_matrix = inputs_list[2]

        query_matrix = tf.linalg.matmul(pre_query_matrix, self.Wq)
        key_matrix = tf.linalg.matmul(pre_key_matrix, self.Wk)
        value_matrix = tf.linalg.matmul(pre_value_matrix, self.Wv)

        attention_matrix = tf.linalg.matmul(query_matrix, key_matrix, transpose_b=True)
        attention_matrix = attention_matrix / self.scaling_term

        attention_matrix = attention_matrix + self.masking_matrix

        attention_matrix = tf.nn.softmax(attention_matrix, axis=-1)

        values_with_context = tf.linalg.matmul(attention_matrix, value_matrix)

        return values_with_context

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'query_dimensions': self.query_dimensions,
                'value_dimensions': self.value_dimensions,
                'use_masking': self.use_masking,
                'num_timesteps': self.num_timesteps,
                'encoding_dim': self.encoding_dim}
