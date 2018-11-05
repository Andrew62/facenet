import tensorflow as tf
from functools import partial
from classifier.preprocess import read_one_image


def read_images(image_buffers_ph, image_shape, is_training_ph):
    # do this so we can change behavior
    read_one_train = partial(read_one_image,
                             is_training=True,
                             image_shape=image_shape)
    read_one_test = partial(read_one_image,
                            is_training=False,
                            image_shape=image_shape)
    return tf.cond(is_training_ph,
                   true_fn=lambda: tf.map_fn(read_one_train, image_buffers_ph, dtype=tf.float32),
                   false_fn=lambda: tf.map_fn(read_one_test, image_buffers_ph, dtype=tf.float32))