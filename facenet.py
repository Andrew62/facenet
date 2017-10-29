import helper
import numpy as np
import tensorflow as tf
from functools import partial
from data import read_one_image
from tensorflow.contrib import slim
from networks import inception_resnet_v2
from loss import loss_func


class FaceNet(object):
    def __init__(self,
                 input_faces_file,
                 checkpoint_dir,
                 batch_size=64,
                 emebedding_size=128,
                 is_training=tf.placeholder(tf.bool, []),
                 starting_learning_rate=0.01,
                 image_shape=(160, 160, 3),
                 identities_per_batch = 40,
                 n_images_per_iden=25,
                 n_validation_images=3000,
                 eval_threholds=np.arange(0, 4, 0.1)):
        self.input_faces_file = input_faces_file
        helper.check_dir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.embedding_size = emebedding_size
        self.is_training = is_training
        self.learning_rate = starting_learning_rate
        self.image_shape = image_shape
        self.identities_per_batch = identities_per_batch
        self.n_images_per_iden = n_images_per_iden
        self.n_val_images = n_validation_images
        self.eval_thresholds = eval_threholds

        image_paths_ph = tf.placeholder(tf.string)

        images = tf.map_fn(read_func, image_paths_ph, dtype=tf.float32)
        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            prelogits, endpoints = inception_resnet_v2.inception_resnet_v2(images,
                                                                           is_training=is_training,
                                                                           num_classes=self.embedding_size)
        self.embeddings = tf.nn.l2_normalize(prelogits, 0, name="embeddings")

        # don't run this until we've already done a batch pass of faces. We
        # compute the triplets offline, stack, and run through again
        anchors, positives, negatives = tf.unstack(tf.reshape(self.embeddings, [-1, 3, self.embedding_size]), axis=1)
        losses = loss_func(anchors, positives, negatives)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([losses] + regularization_loss, name="total_loss")
        learning_rate = tf.train.exponential_decay()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
        tf.summary.scalar("Total Loss", total_loss)

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(checkpoint_dir)
