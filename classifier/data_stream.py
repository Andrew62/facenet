from typing import List, Tuple
import tensorflow as tf
from preprocess import read_one_image


def make_dataset(filenames: List[str],
                 image_size: Tuple(int, int, int)=(224, 224, 3),
                 batch_size: int=32,
                 epochs: int=None) -> tf.data.Iterator:
    record_defaults = [tf.int32, tf.string]

    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)\
        .map(lambda x: read_one_image(x, image_size=image_size)) \
        .shuffle(buffer_size=10000)\
        .repeat(epochs)\
        .batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator

