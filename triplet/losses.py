import tensorflow as tf


def unstack_triplets(tensor, dims):
    return tf.unstack(tf.reshape(tensor, dims), 3, 1)


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


def build_loss(args, embeddings):
    dims = [-1, 3, args.embedding_size]
    if args.loss_func == "face_net":
        # don't run this until we've already done a batch pass of faces. We
        # compute the triplets offline, stack, and run through again
        anchors, positives, negatives = unstack_triplets(embeddings, dims)
        return facenet_loss(anchors, positives, negatives)
    elif args.loss_func == "lossless":
        activated_embeddings = tf.nn.sigmoid(embeddings)
        anchors, positives, negatives = unstack_triplets(activated_embeddings, dims)
        return lossless_triple(anchors, positives, negatives, args.embedding_size,
                                      args.embedding_size)
    else:
        raise Exception("{} is not a valid loss function".format(args.loss_func))