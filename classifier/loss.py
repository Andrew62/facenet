import tensorflow as tf


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
