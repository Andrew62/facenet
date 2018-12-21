import tensorflow as tf
from functools import partial
from classifier.preprocess import preprocess_one_image_buffer


def read_one_image(fp, **kwargs):
    buffer = tf.read_file(fp)
    return preprocess_one_image_buffer(buffer, **kwargs)


def read_images(image_buffers_ph, image_shape, is_training_ph):
    # do this so we can change behavior
    read_one_train = partial(read_one_image,
                             is_training=tf.constant(True, dtype=tf.bool),
                             image_shape=image_shape)
    read_one_test = partial(read_one_image,
                            is_training=tf.constant(False, dtype=tf.bool),
                            image_shape=image_shape)
    return tf.cond(is_training_ph,
                   true_fn=lambda: tf.map_fn(read_one_train, image_buffers_ph, dtype=tf.float32),
                   false_fn=lambda: tf.map_fn(read_one_test, image_buffers_ph, dtype=tf.float32))
