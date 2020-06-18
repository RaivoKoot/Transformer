import tensorflow as tf

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, query_dimensions, value_dimensions, encoding_dim=300, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.Wq = self.add_weight(
            shape=(encoding_dim, query_dimensions), initializer='random_normal',
            trainable=True
        )

        self.Wk = self.add_weight(
            shape=(encoding_dim, query_dimensions), initializer='random_normal',
            trainable=True
        )

        self.Wv = self.add_weight(
            shape=(encoding_dim, value_dimensions), initializer='random_normal',
            trainable=True
        )

        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.scaling_term = tf.math.sqrt(tf.constant(query_dimensions, dtype=tf.float32))

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
        attention_matrix = tf.nn.softmax(attention_matrix, axis=1)

        values_with_context = tf.linalg.matmul(attention_matrix, value_matrix)

        return values_with_context
