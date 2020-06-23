import tensorflow as tf
import pickle
from Transformer import Transformer
from PositionalEncoding import PositionalEncoding
from load_data import get_datasets

NUM_ENCODER_BLOCKS = 6
NUM_DECODER_BLOCKS = 6
NUM_DECODER_DIM = 300 # Word Embedding Dimensions
NUM_ENCODER_DIM = 300 # Image Cell Dimensions
NUM_HEADS = 5
QUERY_DIM_LIST_ENCODER = [32] * NUM_HEADS # Query Dimensions for each Head
VALUE_DIM_LIST_ENCODER = [32] * NUM_HEADS # Value Dimensions for each Head
QUERY_DIM_LIST_DECODER = [32] * NUM_HEADS # Query Dimensions for each Head
VALUE_DIM_LIST_DECODER = [32] * NUM_HEADS # Value Dimensions for each Head

NUM_NEURONS_FEEDFORWARD = 1024 # Neurons in the intermediate layer of
                               # each 2-layer feed-forward block in the
                               # encoder and decoder blocks.
DROPOUT_RATE = 0.1
NUM_TIMESTEPS = 51 # After padding, the num_timesteps each sequence has.
                   # Important for the decoder to build the correct
                   # sized masking matrix.

NUM_CLASSES = 5000 # Vocabulary size

transformer = Transformer(
    query_dimensions_list_encoder = QUERY_DIM_LIST_ENCODER,
    value_dimensions_list_encoder = VALUE_DIM_LIST_ENCODER,
    query_dimensions_list_decoder = QUERY_DIM_LIST_DECODER,
    value_dimensions_list_decoder = VALUE_DIM_LIST_DECODER,
    num_heads = NUM_HEADS,
    num_neurons_feedforward = NUM_NEURONS_FEEDFORWARD,
    encoding_dim_encoder = NUM_ENCODER_DIM,
    encoding_dim_decoder = NUM_DECODER_DIM,
    dropout_rate = DROPOUT_RATE,
    num_timesteps = NUM_TIMESTEPS,
    num_classes = NUM_CLASSES,
    num_encoder_blocks = NUM_ENCODER_BLOCKS,
    num_decoder_blocks = NUM_DECODER_BLOCKS,
)


"""
We need to construct the input embedding and positional encoding
part ourself. Luckily, we have already created a word Tokenizer
and a word Embedding layer in preprocess.ipynb.
"""
CNN_REPRESENTATION_DIM = 2048
NUM_ENCODER_INPUT_TIMESTEPS = 16 # Our CNN outputs 4x4xchannels. Flattened,
                                 # this is 16 timesteps. One for each grid cell.

embedding_file = open('Model Component Save Files/embedding_layer.pickle', 'rb')
tokenizer_file = open('Model Component Save Files/tokenizer.pickle', 'rb')

embedding_layer = pickle.load(embedding_file)
tokenizer = pickle.load(tokenizer_file)

def define_model():
    image_representation = tf.keras.layers.Input(shape=[CNN_REPRESENTATION_DIM], dtype=tf.float32)
    encoder_input = tf.keras.layers.Dense(NUM_ENCODER_DIM, activation='relu')(image_representation)
    encoder_input = PositionalEncoding(NUM_ENCODER_INPUT_TIMESTEPS, NUM_ENCODER_DIM)(encoder_input)

    target_token_sequence = tf.keras.layers.Input(shape=[], dtype=tf.int32)
    decoder_input = embedding_layer(target_token_sequence)
    decoder_input = PositionalEncoding(NUM_TIMESTEPS, NUM_DECODER_DIM)(decoder_input)

    vocabulary_distribution = transformer([encoder_input, decoder_input])

    model = tf.keras.Model(inputs=[image_representation, target_token_sequence],
                           outputs=[vocabulary_distribution])

    return model

model = define_model()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule according to the Transformer paper.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

optimizer = tf.keras.optimizers.Adam(CustomSchedule(NUM_DECODER_DIM),
                                     beta_1=0.9, beta_2=0.98, epsilon=1e-9)

"""
Create a loss that masks out padded tokens.
"""
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

"""
Finally, compile and start training.
"""
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['sparse_categorical_accuracy'])

checkpoint_filepath = 'model_checkpoints/transformer.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_freq='epoch',
    save_best_only=True,
    save_weights_only=True)

train_dataset, val_dataset = get_datasets()

model.fit(train_dataset, epochs=1000,
          validation_data=(val_dataset), callbacks=[model_checkpoint_callback])
