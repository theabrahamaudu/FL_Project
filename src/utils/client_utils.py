"""
Utility functions for client module
"""

import tensorflow as tf

def batch_data(data_shard, bs=32) -> tf.data.Dataset:
    """Takes in a clients data shard and create a TensorFlow Dataset object off it
    Args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    Returns:
        tfds object

    Args:
        data_shard (list): a data, label constituting a client's data shard
        bs (int, optional): Batch Size. Defaults to 32.

    Returns:
        tf.data.Dataset: Shuffled TensorFlow Dataset
    """
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)