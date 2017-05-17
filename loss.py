
import tensorflow as tf


def loss_func(anchors, positives, negatives, alpha=0.2):
    """
    Loss function pulled form FaceNet paper. Loss function is
        sum([||f(anchor) - f(positive))||**2] - [||f(anchor) - f(negative)||**2] + alpha)
    """
    positive_dist = tf.pow(tf.norm(anchors - positives, axis=0, ord=2), 2)
    negative_dist = tf.pow(tf.norm(anchors - negatives, axis=0, ord=2), 2)
    return tf.maximum(tf.reduce_sum(positive_dist - negative_dist + alpha), 0)

