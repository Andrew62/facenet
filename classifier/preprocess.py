import tensorflow as tf


def read_one_image(class_id, fp, **kwargs):
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
    return class_id, tf.image.per_image_standardization(image)