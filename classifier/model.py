import os
import time
import shutil
import numpy as np
import tensorflow as tf
from utils import helper
from tensorflow.contrib import slim
from .data_stream import make_dataset
from .classification_args import ClassificationArgs
from .loss import center_loss as cls_center_loss
from tensorflow.contrib.slim.python.slim.nets import inception_v3


def save_train_params(args: ClassificationArgs):
    print("Parameters:")
    with open(os.path.join(args.checkpoint_dir, "training_params.txt"), 'w') as target:
        for prop in dir(args):
            if not prop.startswith("_"):
                line = "{0}\t{1}".format(prop, getattr(args, prop))
                print(line)
                target.write(line + '\n')


def train(args: ClassificationArgs):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    steps_per_epoch = helper.get_steps_per_epoch([args.train_csv], batch_size=args.batch_size, header=False)
    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        dataset_iterator = make_dataset([args.train_csv], image_size=args.image_shape, batch_size=args.batch_size)
        labels, images = dataset_iterator.get_next()

        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.Variable(0)

        tf.summary.image("input", images, max_outputs=3)

        # do the network thing here
        with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=args.regularization_beta)):
            network_features, _ = inception_v3.inception_v3(images,
                                                            is_training=is_training_ph,
                                                            num_classes=args.embedding_size,
                                                            dropout_keep_prob=args.drop_out)

            # using this val b/c it's what tf slim uses by default
            prelogits_reg = tf.nn.l2_loss(network_features) * args.regularization_beta
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_reg)

            logits = slim.fully_connected(network_features, args.num_classes, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())

        embeddings = tf.nn.l2_normalize(network_features, axis=1, name="l2_embedding")
        center_loss, face_centers = cls_center_loss(embeddings, labels, args.center_loss_alpha, args.num_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, center_loss * args.regularization_beta)
        # one_hot = tf.one_hot(labels, args.num_classes, on_value=1, off_value=0)
        pred = tf.argmax(logits, axis=1, name='predictions')
        class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add_n([class_loss, regularization_loss], name="total_loss")
        # total_loss = tf.add_n([class_loss, center_loss * args.regularization_beta], name="total_loss")

        # decay every 2 epochs
        lr_decay = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=2 * steps_per_epoch,
                                              decay_rate=0.94)

        # explicitly calculate gradients to use clipping per inceptionv3 paper
        original_optimizer = tf.train.RMSPropOptimizer(lr_decay, decay=0.9, epsilon=1.0)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.5)
        train_op = optimizer.minimize(total_loss)

        global_step_inc = tf.assign_add(global_step, 1)

        with tf.name_scope("train"):
            train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))

        tf.summary.scalar("Classification Accuracy", train_acc)
        tf.summary.scalar("Regularization_loss", regularization_loss)
        tf.summary.scalar("Total_loss", total_loss)
        tf.summary.scalar("Softmax_loss", class_loss)
        tf.summary.scalar("Center_loss", center_loss)
        tf.summary.scalar("prelogits_l2_loss", prelogits_reg)
        tf.summary.scalar("learning_rate", lr_decay)
        tf.summary.histogram("centers_hist", face_centers)
        tf.summary.histogram("l2_embeddings", embeddings)
        tf.summary.histogram("network_features", network_features)

        merged_summaries = tf.summary.merge_all()
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

    print("Starting session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config).as_default() as sess:
        summary_writer = tf.summary.FileWriter(args.checkpoint_dir)

        var_list = graph.get_collection("variables")
        saver = tf.train.Saver(var_list=var_list)
        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print("Restoring from " + latest_checkpoint)
            saver.restore(sess, latest_checkpoint)
            sess.run([local_init, dataset_iterator.initializer])
        else:
            print("Initializing!")
            sess.run([global_init, local_init, dataset_iterator.initializer])
        start = time.time()
        try:
            while True:
                try:
                    feed_dict = {
                        is_training_ph: True
                    }
                    if global_step.eval() % 100 == 0:
                        ops_to_run = [merged_summaries, train_op, total_loss, global_step_inc]
                        summary, _, loss, _ = sess.run(ops_to_run, feed_dict=feed_dict)

                        summary_writer.add_summary(summary, global_step.eval())
                        batch_per_sec = (time.time() - start) / global_step.eval()
                        print("model: {0}\tglobal step: {1:,}\t".format(os.path.basename(args.checkpoint_dir),
                                                                        global_step.eval()),
                              "loss: {0:0.5f}\tstep/sec: {1:0.2f}".format(loss, batch_per_sec))
                    else:
                        ops_to_run = [global_step_inc, train_op, total_loss]
                        _, _, loss = sess.run(ops_to_run, feed_dict=feed_dict)

                        if loss == np.inf:  # esta no bueno!
                            raise ValueError("Loss is inf")

                    if ((global_step.eval() + 1) % (args.save_every * steps_per_epoch)) == 0:
                        print("Check pointing")
                        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step.eval())
                except tf.errors.OutOfRangeError:
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting loop")
        except tf.errors.ResourceExhaustedError as e:
            print("Resouce exhausted. try again.")
            shutil.rmtree(args.checkpoint_dir)
            raise e
        print("Training complete. Saving")
        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step.eval())
        print("Done")

