import performance
import numpy as np
import tensorflow as tf
from functools import partial
from data import read_one_image
from tensorflow.contrib import slim
from networks import inception_resnet_v2


class FaceNet(object):
    def __init__(self,
                 image_paths_ph,
                 is_training_ph,
                 embedding_size,
                 global_step_ph,
                 init_learning_rate,
                 image_shape,
                 decay_steps,
                 decay_rate):
        self.is_training_ph = is_training_ph
        self.global_step_ph = global_step_ph
        self.image_paths_ph = image_paths_ph

        # do this so we can change behavior
        read_one_train = partial(read_one_image,
                                 is_training=True,
                                 image_shape=image_shape)
        read_one_test = partial(read_one_image,
                                is_training=False,
                                image_shape=image_shape)
        images = tf.cond(is_training_ph,
                         true_fn=lambda: tf.map_fn(read_one_train, self.image_paths_ph, dtype=tf.float32),
                         false_fn=lambda: tf.map_fn(read_one_test, self.image_paths_ph, dtype=tf.float32))

        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            prelogits, endpoints = inception_resnet_v2.inception_resnet_v2(images,
                                                                           is_training=self.is_training_ph,
                                                                           num_classes=embedding_size,
                                                                           dropout_keep_prob=1.0)
        self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name="l2_embedding")

        # don't run this until we've already done a batch pass of faces. We
        # compute the triplets offline, stack, and run through again
        anchors, positives, negatives = tf.unstack(tf.reshape(self.embeddings, [-1, 3, embedding_size]), 3, 1)
        triplet_loss = self.facenet_loss(anchors, positives, negatives)
        tf.summary.scalar("Triplet_Loss", triplet_loss)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([triplet_loss] + regularization_loss, name="total_loss")
        learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                   decay_rate=decay_rate,  # so far the best run set this to 0.96
                                                   decay_steps=decay_steps,
                                                   staircase=True,
                                                   global_step=self.global_step_ph)
        tf.summary.scalar("Learning_Rate", learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.total_loss)
        tf.summary.scalar("Total_Loss", self.total_loss)
        self.merged_summaries = tf.summary.merge_all()

    @staticmethod
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

    @staticmethod
    def center_loss(embeddings, centers, alpha=0.5):
        """
        https://ydwen.github.io/papers/WenECCV16.pdf

        lambda/2 * sum(||x_i - cy_i||**2)
        """
        # reduce mean here for the same reason as above
        return alpha * tf.reduce_mean(tf.pow(tf.subtract(embeddings, centers), 2))

    @staticmethod
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

    def evaluate(self,
                 sess,
                 validation_set,
                 global_step,
                 batch_size=64,
                 thresholds=np.arange(0, 4, 0.01)):
        col0 = validation_set[:, 0]
        col1 = validation_set[:, 1]
        # TODO is it more efficient to only run unique fps then reassemble?
        all_fps = np.hstack([col0, col1])
        embeddings = self.inference(sess,
                                    all_fps,
                                    batch_size,
                                    False,
                                    global_step)
        n_rows = validation_set.shape[0]
        col0_embeddings = embeddings[:n_rows, :]
        col1_embeddings = embeddings[n_rows:, :]
        l2_dist = self.l2_squared_distance(col0_embeddings, col1_embeddings, axis=1)
        true_labels = validation_set[:, -1].astype(np.int)
        threshold, acc = performance.optimal_threshold(l2_dist, true_labels, thresholds=thresholds)
        pred = np.where(l2_dist <= threshold, 1, 0)
        precision, recall, f1 = performance.precision_recall_f1(pred, true_labels)
        return threshold, acc, precision, recall, f1

    @staticmethod
    def l2_squared_distance(a, b, axis=None):
        return np.sum(np.power(a - b, 2), axis=axis)

    def get_triplets(self, image_paths, embeddings, class_ids, alpha=0.2):
        unique_ids = np.unique(class_ids)
        out_fps = []

        # do this to make the comparison below to check for the same class
        for class_id in unique_ids:
            class_vectors = embeddings[class_ids == class_id, :]
            class_fps = image_paths[class_ids == class_id]
            assert class_vectors.shape[0] == class_fps.shape[0], "embedding and file name mismatch"
            out_of_class_vectors = embeddings[class_ids != class_id, :]
            out_of_class_fps = image_paths[class_ids != class_id]
            assert out_of_class_vectors.shape[0] == out_of_class_fps.shape[0], "embedding and file name mismatch"
            for anchor_idx in range(class_vectors.shape[0] - 1):
                anchor_vec = class_vectors[anchor_idx, :]
                other_positive = class_vectors[anchor_idx + 1:, :]
                pos_dist_values = self.l2_squared_distance(anchor_vec, other_positive, 1)
                pos_idx = np.argmax(pos_dist_values)
                positive_dist = pos_dist_values[pos_idx]
                neg_dist_values = self.l2_squared_distance(class_vectors[anchor_idx, :], out_of_class_vectors, axis=1)
                # np.where returns a list. we want the first element
                valid_negatives = np.where(neg_dist_values - positive_dist < alpha)[0]
                if len(valid_negatives) > 0:
                    # input is a list of indicies
                    neg_idx = np.random.choice(valid_negatives)
                    out_fps.extend([class_fps[anchor_idx], class_fps[pos_idx], out_of_class_fps[neg_idx]])
        print("Generated {0:,} triplets".format(len(out_fps) // 3))
        return np.asarray(out_fps)

    def inference(self,
                  sess,
                  file_paths,
                  batch_size,
                  is_training,
                  global_step):
        embeddings = []
        for idx in range(0, file_paths.shape[0], batch_size):
            batch = file_paths[idx: idx + batch_size]
            embeddings.append(sess.run(self.embeddings, feed_dict={
                self.image_paths_ph: batch,
                self.is_training_ph: is_training,
                self.global_step_ph: global_step
            }))
        return np.vstack(embeddings)
