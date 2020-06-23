import numpy as np
import tensorflow as tf

# Load numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

def load_tfrecord_dataset(filename, PATH='../data/TFRecords/'):
    dataset = tf.data.TFRecordDataset([PATH + filename],
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(5000)
    # Decode the tfrecord protobuf Examples back to image name string tensors
    # and image caption token tensors
    dataset = dataset.map(decode_tfrecord,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # For each image name, load the corresponding image representation
    # from the numpy files
    dataset = dataset.map(lambda img_name, caption: tf.numpy_function(
                map_func, [img_name, caption], [tf.float32, tf.int32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

def decode_tfrecord(tfrecord):
    FEATURE_DESCRIPTIONS = {
        'img_name': tf.io.FixedLenFeature([], tf.string),
        'caption': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(tfrecord, FEATURE_DESCRIPTIONS)

    img_name = tf.io.parse_tensor(example['img_name'], out_type=tf.string)
    caption = tf.io.parse_tensor(example['caption'], out_type=tf.int32)

    return img_name, caption

def setup_for_transformer(dataset):
    """
    Splits the training instance into 3 parts: Encoder input, Decoder input,
    and target sequence.
    """
    return dataset.map(lambda image, caption_sequence:
                        ((image, caption_sequence[:51]), caption_sequence[1:52]))

def get_datasets():
    train = load_tfrecord_dataset('train.tfrecord')
    val = load_tfrecord_dataset('val.tfrecord')
    return setup_for_transformer(train), setup_for_transformer(val)
