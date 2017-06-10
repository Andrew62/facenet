
import numpy as np
import tensorflow as tf


def l2_squared_distance(a, b, axis=None):
    return np.sum(np.power(a - b, 2), axis=axis)


def loss_func(anchors, positives, negatives, alpha=0.2):
    """
    Loss function pulled form FaceNet paper. Loss function is
        sum([||f(anchor) - f(positive))||**2] - [||f(anchor) - f(negative)||**2] + alpha)
    """
    # reduce row-wise
    positive_dist = tf.reduce_sum(tf.pow(anchors - positives, 2), 1)
    tf.summary.scalar("positive_distance", tf.reduce_sum(positive_dist))
    negative_dist = tf.reduce_sum(tf.pow(anchors - negatives, 2), 1)
    tf.summary.scalar("negative_distance", tf.reduce_sum(negative_dist))
    loss = positive_dist - negative_dist + alpha
    # reduce mean since we could have variable batch size
    return tf.reduce_mean(tf.maximum(loss, 0.0), 0)
