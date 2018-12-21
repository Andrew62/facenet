from typing import List, Tuple
import tensorflow as tf
from classifier.preprocess import read_one_image_fp


def make_dataset(filenames: List[str],
                 image_size: Tuple[int, int, int]=(224, 224, 3),
                 batch_size: int=32,
                 epochs: int=None,
                 is_training: bool=False) -> tf.data.Iterator:
    record_defaults = [tf.int64, tf.string]

    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)\
        .map(lambda cid, fp: read_one_image_fp(cid, fp, image_size=image_size, is_training=is_training)) \
        .shuffle(buffer_size=10000)\
        .repeat(epochs)\
        .batch(batch_size)
    return dataset.make_initializable_iterator()

