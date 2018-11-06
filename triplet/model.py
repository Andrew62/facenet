
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from triplet.data import Dataset
from tensorflow.contrib import slim
from triplet import train_ops, losses
from triplet.params import ModelParams
from triplet.preprocess import read_images
from tensorflow.contrib.slim.python.slim.nets import inception_v3


def save_train_params(args: ModelParams):
    print("Parameters:")
    with open(os.path.join(args.checkpoint_dir, "training_params.txt"), 'w') as target:
        for prop in dir(args):
            if not prop.startswith("_"):
                line = "{0}\t{1}".format(prop, getattr(args, prop))
                print(line)
                target.write(line + '\n')


def model_train(args: ModelParams):
    thresholds = np.arange(0, 4, 0.1)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    dataset = Dataset(args.input_faces,
                      n_identities_per=args.identities_per_batch,
                      n_images_per=args.n_images_per_iden)
    save_train_params(args)

    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        image_paths_ph = tf.placeholder(tf.string, name="input_image_buffers")
        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.Variable(0)

        images = read_images(image_paths_ph, args.image_shape, is_training_ph)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=args.regularization_beta)):
            network_features, _ = inception_v3.inception_v3(images,
                                                            is_training=is_training_ph,
                                                            num_classes=args.embedding_size,
                                                            dropout_keep_prob=args.drop_out)

        embeddings = tf.nn.l2_normalize(network_features, axis=1, name="l2_embedding")
        triplet_loss = losses.build_loss(args, embeddings)
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add_n([triplet_loss, regularization_loss], name="total_loss")

        # decay every 2 epochs
        lr_decay = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=2 * dataset.total_images,
                                              decay_rate=0.94)

        # explicitly calculate gradients to use clipping per inceptionv3 paper
        original_optimizer = tf.train.RMSPropOptimizer(lr_decay, decay=0.9, epsilon=1.0)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.5)
        train_op = optimizer.minimize(total_loss)
        global_step_inc = tf.assign_add(global_step, 1)

        # summaries!
        tf.summary.image("input", images, max_outputs=3)
        tf.summary.scalar("Triplet_Loss", triplet_loss)
        tf.summary.scalar("Total loss", total_loss)
        tf.summary.scalar("Regularization_loss", regularization_loss)
        tf.summary.scalar("learning_rate", lr_decay)
        tf.summary.histogram("normed_embeddings", embeddings)

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
            sess.run(local_init)
        else:
            print("Initializing!")
            sess.run([global_init, local_init])

        start = time.time()
        lfw = Dataset(args.lfw, n_eval_pairs=args.n_validation)

        print("Starting loop")
        while global_step.eval() < args.train_steps:
            try:
                # embed and collect all current face weights
                image_paths, labels = dataset.get_train_batch()
                embeddings_np = train_ops.inference(image_paths, sess, embeddings, image_paths_ph,
                                                    is_training_ph, args.batch_size)
                if np.any(np.isnan(embeddings_np)) or np.any(np.isinf(embeddings_np)):
                    print("NaNs or inf found in embeddings. Exiting")
                    raise ValueError("NaNs or inf found in embeddings")

                triplets, _ = train_ops.get_triplets(image_paths, embeddings_np, labels)
                n_trips = triplets.shape[0]
                trip_step = args.batch_size - (args.batch_size % 3)
                for idx in range(0, n_trips, trip_step):
                    feed_dict = {
                        image_paths_ph: triplets[idx: idx + trip_step],
                        is_training_ph: True,
                    }
                    # should one batch through triplets be one step or should it be left like this?
                    if global_step.eval() % 100 == 0:
                        summary, _, loss, _ = sess.run([merged_summaries, train_op, total_loss, global_step_inc],
                                                       feed_dict=feed_dict)
                        batch_per_sec = (time.time() - start) / global_step.eval()
                        summary_writer.add_summary(summary, global_step.eval())
                        print("model: {0}\tglobal step: {1:,}\t".format(os.path.basename(args.checkpoint_dir),
                                                                        global_step.eval()),
                              "loss: {0:0.5f}\tstep/sec: {1:0.2f}".format(loss, batch_per_sec))
                    else:
                        _, loss, _ = sess.run([train_op, total_loss, global_step_inc], feed_dict=feed_dict)

                    if loss == np.inf:  # esta no bueno!
                        raise ValueError("Loss is inf: {}".format(loss))

                    if global_step.eval() % 10000 == 0:
                        saver.save(sess, os.path.join(args.checkpoint_dir, 'triplet'), global_step=global_step.eval())
                        start = time.time()
                        evaluation_set = lfw.get_evaluation_batch()
                        threshold, accuracy = train_ops.evaluate(sess, evaluation_set, embeddings,
                                                                 image_paths_ph, is_training_ph,
                                                                 args.batch_size, thresholds)
                        eval_summary = tf.Summary()
                        eval_summary.value.add(tag="lfw_verification_accuracy", simple_value=accuracy)
                        eval_summary.value.add(tag="lfw_verification_threshold", simple_value=threshold)
                        summary_writer.add_summary(eval_summary, global_step=global_step.eval())
                        print("eval @{0:,} steps\taccuracy: {1:0.2%}\tthreshold: {2:0.2f}".format(global_step.eval(),
                                                                                                  accuracy, threshold))
            except KeyboardInterrupt:
                print("Keyboard Interrupt. Exiting.")
                break
            except tf.errors.ResourceExhaustedError as e:
                print("Resouce exhausted. try again.")
                shutil.rmtree(args.checkpoint_dir)
                raise e
        print("Saving model...")
        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet'), global_step=global_step.eval())
        print("Saved to: {}".format(args.checkpoint_dir))
    print("Done")
