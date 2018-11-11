import tensorflow as tf


def preprocess_one_image_buffer(buffer, **kwargs):
    """
    Preprocess one image. Decode an image buffer (png|jpg|gif), reshape, random flip, and standardize
    :param buffer: string representing an image
    :param image_shape: three tuple for image shape. default (224, 224, 3)
    :param is_training: boolean. default True
    :return: processed image tensor
    """
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    is_training = kwargs.pop("is_training", False)
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


def read_one_image_fp(class_id, fp, **kwargs):
    """
    Read one image given a file path and class id
    :param class_id: unique key representing the image category
    :param fp: string path to jpg|png|gif to load
    :param image_shape: three tuple for image shape. default (224, 224, 3)
    :param is_training: boolean. default True
    :return: class_id, processed image tensor
    """
    buffer = tf.read_file(fp)
    return class_id, preprocess_one_image_buffer(buffer, **kwargs)
