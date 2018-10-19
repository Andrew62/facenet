import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from center_loss import center_loss
from data_stream import make_dataset
from networks import inception_resnet_v2
from classification_args import ClassificationArgs


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
    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        dataset_iterator = make_dataset([args.train_csv], image_shape=args.image_shape)
        labels, images = dataset_iterator.get_next(0)

        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.Variable(0)

        tf.summary.image("input", images, max_outputs=3)

        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            network_features, _ = inception_resnet_v2.inception_resnet_v2(images,
                                                                          is_training=is_training_ph,
                                                                          num_classes=args.embedding_size,
                                                                          dropout_keep_prob=args.drop_out)

            # using this val b/c it's what tf slim uses by default
            prelogits_reg = tf.nn.l2_loss(network_features) * 4e-5
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_reg)

            logits = slim.fully_connected(network_features, args.num_classes, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())

        embeddings = tf.nn.l2_normalize(network_features, axis=1, name="l2_embedding")
        center_loss, face_centers = center_loss(embeddings, labels, args.center_loss_alpha, args.num_classes)
        one_hot = tf.one_hot(labels, args.num_classes, on_value=1, off_value=0)
        pred = tf.argmax(logits, axis=1, name='predictions')
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logits))

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([class_loss, center_loss] + regularization_loss, name="total_loss")

        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)
        global_step_inc = tf.assign_add(global_step, 1)

        with tf.name_scope("train"):
            train_acc, train_acc_op = tf.metrics.accuracy(one_hot, pred)

        tf.summary.scalar("Classification Accuracy", train_acc)
        tf.summary.scalar("Total_loss", total_loss)
        tf.summary.scalar("Softmax_loss", class_loss)
        tf.summary.scalar("Center_loss", center_loss)
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
                        ops_to_run = [merged_summaries, optimizer, total_loss, global_step_inc, train_acc_op]
                        summary, _, loss, _,  = sess.run(ops_to_run, feed_dict=feed_dict)

                        summary_writer.add_summary(summary, global_step)
                        batch_per_sec = (time.time() - start) / global_step.eval()
                        print("model: {0}\tglobal step: {1:,}\t".format(os.path.basename(args.checkpoint_dir),
                                                                        global_step.eval()),
                              "loss: {0:0.5f}\tstep/sec: {1:0.2f}".format(loss, batch_per_sec))
                    else:
                        ops_to_run = [global_step_inc, train_acc_op, optimizer, total_loss]
                        _, _, _, loss = sess.run(ops_to_run, feed_dict=feed_dict)

                        if loss == np.inf:  # esta no bueno!
                            raise ValueError("Loss is inf")

                    if ((global_step.eval() + 1) % args.save_every) == 0:
                        print("Check pointing")
                        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step)
                except tf.errors.OutOfRangeError:
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting loop")
        except tf.errors.ResourceExhaustedError as e:
            print("Resouce exhausted. try again.")
            shutil.rmtree(args.checkpoint_dir)
            raise e
    print("Training complete. Saving")
    saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step)
    print("Done")


def main():
    args = ClassificationArgs(epochs=90,
                              checkpoint_dir="checkpoints/softmax/" + "2018-05-20-1418",  #helper.get_current_timestamp(),
                              save_every=3000,
                              embedding_size=256,
                              train_csv="fixtures/youtube_subset.csv",
                              batch_size=64,
                              learning_rate=0.01,
                              image_shape=(160, 160, 3))
    train(args)


if __name__ == "__main__":
    main()
