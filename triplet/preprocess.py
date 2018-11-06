import tensorflow as tf
from functools import partial
from classifier.preprocess import read_one_image


def read_one_image(fp, **kwargs):
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    is_training = kwargs.pop("is_training", False)

    buffer = tf.read_file(fp)

    # decode buffer as an image
    image = tf.image.decode_image(buffer, channels=img_shape[-1])

    image = tf.image.resize_bicubic(tf.expand_dims(image, 0), (img_shape[0], img_shape[1]))
    image = tf.squeeze(image, [0])
    if is_training:
        up_down = tf.random_uniform([], minval=0, maxval=1)
        image = tf.cond(up_down > 0.5,
                        lambda: tf.image.flip_up_down(image),
                        lambda: tf.image.random_flip_left_right(image))
    image.set_shape(img_shape)
    return tf.image.per_image_standardization(image)


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
