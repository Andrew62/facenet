import numpy as np
import tensorflow as tf


def loss_func(anchors, positives, negatives, alpha=0.2):
    """
    Loss function pulled form FaceNet paper. Loss function is
        sum([||f(anchor) - f(positive))||**2] - [||f(anchor) - f(negative)||**2] + alpha)
    """
    positive_dist = tf.pow(tf.norm(anchors - positives, axis=1, ord=2), 2)
    negative_dist = tf.pow(tf.norm(anchors - negatives, axis=1, ord=2), 2)
    return tf.reduce_sum(positive_dist - negative_dist + alpha)


def get_negatives(anchors, positives, class_ids):
    samples = anchors.shape[0]
    out_negatives = np.zeros(shape=anchors.shape, dtype=anchors.dtype)
    all_vectors = np.vstack([anchors, positives])
    # do this to make the comparison below to check for the same class
    all_class_ids = np.concatenate([class_ids, class_ids])
    for idx in range(samples):
        current_vect = anchors[idx, :]
        current_compare = np.copy(all_vectors)
        # set all these to the inf because we want the closest
        # negatives
        current_compare[idx, :] = np.inf
        current_compare[idx + samples, :] = np.inf
        current_compare[class_ids[idx] == all_class_ids, :] = np.inf
        dist = np.power(np.linalg.norm(current_vect - current_compare, axis=1), 2)
        out_negatives[idx, :] = all_vectors[np.argmin(dist), :]
    return out_negatives
