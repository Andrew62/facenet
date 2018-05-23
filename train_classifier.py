import os
import time
import shutil
import losses
import numpy as np
import tensorflow as tf
from utils import sprite
from utils import helper
from functools import partial
from data import read_one_image
from tensorflow.contrib import slim
from networks import inception_resnet_v2
from tensorflow.contrib.tensorboard.plugins import projector


class ClassificationArgs(object):
    def __init__(self, **kwargs):
        self.checkpoint_dir = kwargs.pop("checkpoint_dir")
        self.train_csv = kwargs.pop("train_csv")
        self.train_idx_to_name = kwargs.pop("train_idx_to_name")
        self.lfw_csv = kwargs.pop("lfw_csv")
        self.lfw_idx_to_name = kwargs.pop("lfw_idx_to_name")
        self.batch_size = kwargs.pop("batch_size", 32)
        self.image_shape = kwargs.pop("image_shape", (299, 299, 3))
        self.drop_out = kwargs.pop("drop_out", 0.8)
        self.center_loss_alpha = kwargs.pop("center_loss_alpha", 0.5)
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.save_every = kwargs.pop('save_every', 10)
        self.epochs = kwargs.pop("epochs", 90)
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.eval_every = kwargs.pop('eval_every', 10)
        self.decay_epochs = kwargs.pop("decay_epochs", 5)
        self.decay_rate = kwargs.pop('decay_rate', 0.96)


def read_train_csv(fp: str) -> np.array:
    class_ids = np.genfromtxt(fp, delimiter=',', usecols=(0,), dtype=int)
    image_fps = np.genfromtxt(fp, delimiter=',', usecols=(1,), dtype=str)
    return np.array([class_ids, image_fps], dtype='O').T


def process_all_images(sess, global_step, lfw, image_ph, embeddings_op, args, is_training_ph):
    embeddings = []
    for idx in range(0, lfw.shape[0], args.batch_size):
        batch = lfw[idx: idx + args.batch_size, :]
        buffers = helper.read_buffer_vect(batch[:, 1])
        embeddings.append(sess.run(embeddings_op, feed_dict={image_ph: buffers, is_training_ph: False}))
    all_embeddings = np.concatenate(embeddings)
    image_ids_np = lfw[:, 0]
    np.savez(os.path.join(args.checkpoint_dir, "embeddings_{0}.npz".format(global_step)),
             embeddings=all_embeddings,
             class_codes=image_ids_np)
    return all_embeddings, image_ids_np


def train(args: ClassificationArgs):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    data = read_train_csv(args.train_csv)
    data_idx_to_name = np.genfromtxt(args.train_idx_to_name, dtype=str, delimiter=',')
    unique_classes = np.unique(data[:, 0])

    # make projector
    thumbnail_size = [25, 25]
    projector_metadata = os.path.join(args.checkpoint_dir, 'metadata.tsv')
    projector_sprite = os.path.join(args.checkpoint_dir, 'faces.png')
    out_fps = []
    out_names = []
    for name_idx in unique_classes:
        out_fps.append(data[data[:, 0] == name_idx][0][1])
        out_names.append(data_idx_to_name[name_idx])
    sprite.sprite_metadata(out_names, projector_metadata)
    sprite.make_sprite(out_fps, projector_sprite, thumbnail_size=thumbnail_size)

    n_classes = unique_classes.shape[0]

    # drop learning rate when we save. converted from epochs
    batches_per_epoch = data.shape[0] // args.batch_size
    print("{:,} batches/epoch".format(batches_per_epoch))
    decay_steps = batches_per_epoch * args.decay_epochs

    lfw = read_train_csv(args.lfw_csv)
    lfw_name_to_idx = np.genfromtxt(args.lfw_idx_to_name, dtype=str, delimiter=',')

    print("Parameters:")
    with open(os.path.join(args.checkpoint_dir, "training_params.txt"), 'w') as target:
        for prop in dir(args):
            if not prop.startswith("_"):
                line = "{0}\t{1}".format(prop, getattr(args, prop))
                print(line)
                target.write(line + '\n')

    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        image_buffers_ph = tf.placeholder(tf.string, name="input_image_buffers")
        labels_ph = tf.placeholder(tf.int32, name="input_labels")
        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        global_step_ph = tf.placeholder(tf.int32, name="global_step")

        read_one_train = partial(read_one_image,
                                 is_training=True,
                                 image_shape=args.image_shape)
        read_one_test = partial(read_one_image,
                                is_training=False,
                                image_shape=args.image_shape)
        images = tf.cond(is_training_ph,
                         true_fn=lambda: tf.map_fn(read_one_train, image_buffers_ph, dtype=tf.float32),
                         false_fn=lambda: tf.map_fn(read_one_test, image_buffers_ph, dtype=tf.float32))
        tf.summary.image("input", images, max_outputs=3)

        drop_out_keep = tf.cond(is_training_ph,
                                true_fn=lambda: args.drop_out,
                                false_fn=lambda: 1.0)

        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            network_features, _ = inception_resnet_v2.inception_resnet_v2(images,
                                                                          is_training=is_training_ph,
                                                                          num_classes=args.embedding_size,
                                                                          dropout_keep_prob=drop_out_keep)
            tf.summary.histogram("network_features", network_features)
            # using this val b/c it's what tf slim uses by default
            prelogits_reg = tf.nn.l2_loss(network_features) * 4e-5
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_reg)
            logits = slim.fully_connected(network_features, n_classes, activation_fn=None)
        embeddings = tf.nn.l2_normalize(network_features, 1, 1e-10, name="l2_embedding")
        tf.summary.histogram("l2_embeddings", embeddings)
        predictions = tf.argmax(logits, 1, name="prediction")
        center_loss, face_centers = losses.center_loss(embeddings, labels_ph, args.center_loss_alpha, n_classes)
        tf.summary.scalar("Center_loss", center_loss)
        tf.summary.histogram("centers_hist", face_centers)
        one_hot = tf.one_hot(labels_ph, n_classes, on_value=1, off_value=0)
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=logits))
        tf.summary.scalar("Softmax_loss", class_loss)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([class_loss, center_loss] + regularization_loss, name="total_loss")
        tf.summary.scalar("Total_loss", total_loss)
        learning_rate = tf.train.exponential_decay(args.learning_rate, global_step_ph, decay_steps=decay_steps,
                                                   staircase=True, decay_rate=args.decay_rate)
        tf.summary.scalar("Learning_Rate", learning_rate)
        # adam optimizer set to andrew ng defaults
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7).minimize(total_loss)
        merged_summaries = tf.summary.merge_all()
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

    print("Starting session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    np.random.shuffle(data)
    with tf.Session(graph=graph, config=config).as_default() as sess:
        summary_writer = tf.summary.FileWriter(args.checkpoint_dir,
                                               graph=graph)
        var_list = graph.get_collection("variables")
        saver = tf.train.Saver(var_list=var_list)
        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print("Restoring from " + latest_checkpoint)
            saver.restore(sess, latest_checkpoint)
            global_step = int(latest_checkpoint.split("-")[-1])
            start_epoch = int(global_step // batches_per_epoch)
            sess.run(local_init)
        else:
            print("Initializing!")
            sess.run([global_init, local_init])
            global_step = 0
            start_epoch = 0
        start = time.time()
        try:
            for epoch in range(start_epoch, args.epochs):
                for idx in range(0, data.shape[0], args.batch_size):
                    batch = data[idx: idx + args.batch_size, :]
                    buffers = helper.read_buffer_vect(batch[:, 1])
                    feed_dict = {
                        image_buffers_ph: buffers,
                        labels_ph: batch[:, 0],
                        is_training_ph: True,
                        global_step_ph: global_step
                    }
                    if global_step % 100 == 0:
                        summary, _, loss = sess.run([merged_summaries, optimizer, total_loss], feed_dict=feed_dict)
                        summary_writer.add_summary(summary, global_step)
                        batch_per_sec = (time.time() - start) / global_step
                        print("model: {0}\tepoch: {1:,}\tglobal step: {2:,}\t".format(os.path.basename(args.checkpoint_dir),
                                                                                      epoch, global_step),
                              "loss: {1:0.5f}\tstep/sec: {2:0.2f}".format(global_step, loss, batch_per_sec))
                    else:
                        _, loss = sess.run([optimizer, total_loss], feed_dict=feed_dict)
                    global_step += 1

                    if loss == np.inf:  # esta no bueno!
                        raise ValueError("Loss is inf")
                # shuffle data
                np.random.shuffle(data)

                if ((epoch + 1) % args.save_every) == 0:
                    print("Check pointing")
                    saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step)

                if ((epoch + 1) % args.eval_every) == 0:
                    print("evaluating")
                    all_embeddings, image_ids = process_all_images(sess, global_step, lfw, image_buffers_ph, embeddings,
                                                                   args, is_training_ph)
                    for name in ['andrew', 'erin']:
                        name_to_idx = np.where(lfw_name_to_idx == name)[0][0]
                        person_embed = all_embeddings[image_ids == name_to_idx, :]
                        sim = np.dot(all_embeddings, person_embed[0])
                        sorted_values = np.argsort(sim)[::-1]
                        print("Similar to {0}".format(name.title()))
                        for sidx in sorted_values[1:6]:
                            sv = sim[sidx]
                            # TODO are we just getting high/low values b/c it's early in training?
                            # if np.isnan(sv) or sv == 0 or sv == 1.0:
                            #     raise ValueError("Comparison value is {0}. Aborting".format(sv))
                            print("\t{0} ({1:0.5f})".format(lfw_name_to_idx[image_ids[sidx]], sv))
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting loop")
        except tf.errors.ResourceExhaustedError as e:
            print("Resouce exhausted. try again.")
            shutil.rmtree(args.checkpoint_dir)
            raise e
    print("Training complete. Saving")
    # projector visualization
    prj_config = projector.ProjectorConfig()
    face_centers_prj = prj_config.embeddings.add()
    face_centers_prj.tensor_name = 'centers:0'  # pulled from the variable within losses.py
    face_centers_prj.metadata_path = projector_metadata
    face_centers_prj.sprite.image_path = projector_sprite
    face_centers_prj.sprite.single_image_dim.extend(thumbnail_size)
    projector.visualize_embeddings(summary_writer, prj_config)
    # projector visualization

    saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet_classifier'), global_step=global_step)
    print("Done")


def main():
    args = ClassificationArgs(epochs=270,
                              checkpoint_dir="checkpoints/softmax/" + "2018-05-20-1418",  #helper.get_current_timestamp(),
                              save_every=5,
                              embedding_size=512,
                              lfw_csv="fixtures/lfw.csv",
                              lfw_idx_to_name="fixtures/lfw.csv.classes",
                              train_csv="fixtures/youtube_subset.csv",
                              train_idx_to_name="fixtures/youtube_subset.csv.classes",
                              batch_size=64,
                              learning_rate=0.01,
                              image_shape=(160, 160, 3),
                              decay_rate=0.96,
                              decay_epochs=5)
    train(args)


if __name__ == "__main__":
    main()
