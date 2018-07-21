import tensorflow as tf


def facenet_loss(anchors, positives, negatives, alpha=0.2):
    """
    Loss function pulled form FaceNet paper. Loss function is
        ave([||f(anchor) - f(positive))||**2] - [||f(anchor) - f(negative)||**2] + alpha)
    """
    # reduce row-wise
    positive_dist = tf.reduce_sum(tf.square(anchors - positives), -1)
    tf.summary.scalar("Positive_Distance", tf.reduce_mean(positive_dist))
    negative_dist = tf.reduce_sum(tf.square(anchors - negatives), -1)
    tf.summary.scalar("Negative_Distance", tf.reduce_mean(negative_dist))
    loss = positive_dist - negative_dist + alpha
    # reduce mean since we could have variable batch size
    return tf.reduce_mean(tf.maximum(loss, 0.0), 0)


def center_loss(embeddings, label, alpha, n_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    embed_dim = embeddings.get_shape()[1]
    centers = tf.get_variable('centers', [n_classes, embed_dim], dtype=tf.float32,
                              initializer=tf.zeros_initializer(),
                              trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - embeddings)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(embeddings - centers_batch))
    return loss, centers


def lossless_triple(anchor, positive, negative, n, beta, epsilon=1e-8):
    """Dont' forget to sigmoid

    http://coffeeanddata.ca/lossless-triplet-loss/
    """
    pos = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    pos_dist = -tf.log(-tf.divide(pos, beta) + 1 + epsilon)
    tf.summary.scalar("Positive_Distance", tf.reduce_mean(pos_dist))
    neg_dist = -tf.log(-tf.divide(n - neg, beta) + 1 + epsilon)
    tf.summary.scalar("Negative_Distance", tf.reduce_mean(neg_dist))
    return tf.reduce_mean(neg_dist + pos_dist)
