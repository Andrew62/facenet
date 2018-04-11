import tensorflow as tf


def facenet_loss(anchors, positives, negatives, alpha=0.2):
    """
    Loss function pulled form FaceNet paper. Loss function is
        ave([||f(anchor) - f(positive))||**2] - [||f(anchor) - f(negative)||**2] + alpha)
    """
    # reduce row-wise
    positive_dist = tf.reduce_sum(tf.pow(anchors - positives, 2), 1)
    tf.summary.scalar("Positive_Distance", tf.reduce_mean(positive_dist))
    negative_dist = tf.reduce_sum(tf.pow(anchors - negatives, 2), 1)
    tf.summary.scalar("Negative_Distance", tf.reduce_mean(negative_dist))
    loss = positive_dist - negative_dist + alpha
    # reduce mean since we could have variable batch size
    return tf.reduce_mean(tf.maximum(loss, 0.0), 0)


def center_loss(embeddings, centers, alpha=0.5):
    """
    https://ydwen.github.io/papers/WenECCV16.pdf

    lambda/2 * sum(||x_i - cy_i||**2)
    """
    # reduce mean here for the same reason as above
    return alpha * tf.reduce_mean(tf.pow(tf.subtract(embeddings, centers), 2))


def attalos_loss(anchors, positives, negatives):
    """Pulled from another embedding project: https://github.com/Lab41/attalos
    """

    def meanlogsig(a, b):
        reduction_indices = 2
        return tf.reduce_mean(
            tf.log(tf.sigmoid(tf.reduce_sum(a * b, reduction_indices=reduction_indices))))

    pos_loss = meanlogsig(anchors, positives)
    neg_loss = meanlogsig(-anchors, negatives)
    return -(pos_loss + neg_loss)


def lossless_triple(anchor, positive, negative, n, beta, epsilon=1e-8):
    """Dont' forget to sigmoid

    http://coffeeanddata.ca/lossless-triplet-loss/"""
    pos = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    pos_dist = -tf.log(-tf.divide(pos, beta) + 1 + epsilon)
    tf.summary.scalar("Positive_Distance", tf.reduce_mean(pos_dist))
    neg_dist = -tf.log(-tf.divide(n - neg, beta) + 1 + epsilon)
    tf.summary.scalar("Negative_Distance", tf.reduce_mean(neg_dist))
    return tf.reduce_mean(neg_dist + pos_dist)
