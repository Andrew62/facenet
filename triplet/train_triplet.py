
import os
import time
import losses
import numpy as np
import tensorflow as tf
import triplet_trian_ops
from utils import helper
from functools import partial
from itertools import product
from tensorflow.contrib import slim
from transfer import load_partial_model
<<<<<<< HEAD:train_triplet.py
from networks import inception_resnet_v2
from data import Dataset, read_one_image


class ModelParams(object):
    def __init__(self, **kwargs):
        self.input_faces = kwargs.pop("input_faces", "fixtures/faces.json")
        self.checkpoint_dir = kwargs.pop("checkpoint_dir")
        self.batch_size = kwargs.pop("batch_size", 64)
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.identities_per_batch = kwargs.pop("identities_per_batch", 100)
        self.n_images_per_iden = kwargs.pop("n_images_per_iden", 25)
        self.n_validation = kwargs.pop("n_validation", 1000)
        self.pretrained_base_model = kwargs.pop("pretrained_base_model",
                                                "checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt")
        self.train_steps = kwargs.pop("train_steps", 40000)
        self.lfw = "fixtures/lfw.json"
        self.loss_func = kwargs.pop("loss_func", "lossless")
        self.center_loss_alpha = kwargs.pop("center_loss_alpha", 0.5)
        self.use_center_loss = kwargs.pop("use_center_loss", False)


def model_train(args: ModelParams):
=======
import performance


def process_all_images(dataset, network, sess, global_step, args):
    all_images, image_ids = dataset.get_all_files()
    all_images_np, image_ids_np = np.array(all_images), np.array(image_ids)
    all_embeddings = network.inference(sess,
                                       all_images_np,
                                       args.batch_size,
                                       False,
                                       global_step)
    np.savez(os.path.join(args.checkpoint_dir, "embeddings_{0}.npz".format(global_step)),
             embeddings=all_embeddings,
             class_codes=image_ids_np)
    return all_embeddings, image_ids_np


def model_train(args):
>>>>>>> 9f020b5fd29c1ad87572a4a0708e49516f27e374:triplet/train_triplet.py
    image_shape = (160, 160, 3)
    thresholds = np.arange(0, 4, 0.1)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Parameters:")
    with open(os.path.join(args.checkpoint_dir, "training_params.txt"), 'w') as target:
        for prop in dir(args):
            if not prop.startswith("_"):
                line = "{0}\t{1}".format(prop, getattr(args, prop))
                print(line)
                target.write(line + '\n')

    dataset = Dataset(args.input_faces,
                      n_identities_per=args.identities_per_batch,
                      n_images_per=args.n_images_per_iden)

    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        image_buffers_ph = tf.placeholder(tf.string, name="input_image_buffers")
        global_step_ph = tf.placeholder(tf.int32, name="global_step")
        is_training_ph = tf.placeholder(tf.bool, name="is_training")
        # do this so we can change behavior
        read_one_train = partial(read_one_image,
                                 is_training=True,
                                 image_shape=image_shape)
        read_one_test = partial(read_one_image,
                                is_training=False,
                                image_shape=image_shape)
        images = tf.cond(is_training_ph,
                         true_fn=lambda: tf.map_fn(read_one_train, image_buffers_ph, dtype=tf.float32),
                         false_fn=lambda: tf.map_fn(read_one_test, image_buffers_ph, dtype=tf.float32))
        tf.summary.image("input", images, max_outputs=12)
        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(images,
                                                                is_training=is_training_ph,
                                                                num_classes=args.embedding_size,
                                                                dropout_keep_prob=0.8)

        embeddings = tf.nn.l2_normalize(logits, 1, name="l2_embedding")
        tf.summary.histogram("normed_embeddings", embeddings)

        if args.loss_func == "face_net":
            # don't run this until we've already done a batch pass of faces. We
            # compute the triplets offline, stack, and run through again
            anchors, positives, negatives = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
            triplet_loss = losses.facenet_loss(anchors, positives, negatives)
        elif args.loss_func == "lossless":
            activated_embeddings = tf.nn.sigmoid(embeddings)
            anchors, positives, negatives = tf.unstack(tf.reshape(activated_embeddings,
                                                                  [-1, 3, args.embedding_size]), 3, 1)
            triplet_loss = losses.lossless_triple(anchors, positives, negatives, args.embedding_size,
                                                  args.embedding_size)
        else:
            raise Exception("{} is not a valid loss function".format(args.loss_func))

        tf.summary.scalar("Triplet_Loss", triplet_loss)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_loss, name="total_loss")
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)
        tf.summary.scalar("Total loss", total_loss)
        merged_summaries = tf.summary.merge_all()
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

    print("Starting session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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
            sess.run(local_init)
        else:
            print("Initializing!")
            sess.run([global_init, local_init])
            global_step = 0

        start = time.time()

        lfw = Dataset(args.lfw, n_eval_pairs=args.n_validation)

        # write this to disc early in case we want to inspect embedding checkpoints
        helper.to_json(lfw.idx_to_name, os.path.join(args.checkpoint_dir, "idx_to_name.json"))
        print("Starting loop")
        while global_step < args.train_steps:
            try:
                # embed and collect all current face weights
                image_paths, classes = dataset.get_train_batch()
                embeddings_np = triplet_trian_ops.inference(image_paths, sess, embeddings, image_buffers_ph,
                                                            is_training_ph, global_step_ph, args.batch_size,
                                                            global_step)
                if np.any(np.isnan(embeddings_np)) or np.any(np.isinf(embeddings_np)):
                    print("NaNs or inf found in embeddings. Exiting")
                    raise ValueError("NaNs or inf found in embeddings")

                triplets = triplet_trian_ops.get_triplets(image_paths, embeddings_np, classes)
                n_trips = triplets.shape[0]
                trip_step = args.batch_size - (args.batch_size % 3)
                for idx in range(0, n_trips, trip_step):
                    trip_buffers = helper.read_buffer_vect(triplets[idx: idx + trip_step])
                    feed_dict = {
                        image_buffers_ph: trip_buffers,
                        is_training_ph: True,
                        global_step_ph: global_step
                    }
                    # should one batch through triplets be one step or should it be left like this?
                    global_step += 1
                    batch_per_sec = (time.time() - start) / global_step
                    if global_step % 100 == 0:
                        summary, _, loss = sess.run([merged_summaries, optimizer, total_loss], feed_dict=feed_dict)
                        summary_writer.add_summary(summary, global_step)
                        print("model: {0}\tlobal step: {1:,}\t".format(os.path.basename(args.checkpoint_dir),
                                                                       global_step),
                              "loss: {1:0.5f}\tstep/sec: {2:0.2f}".format(global_step, loss, batch_per_sec))
                    else:
                        _, loss = sess.run([optimizer, total_loss], feed_dict=feed_dict)

                    if global_step % 1000 == 0:
                        saver.save(sess, args.checkpoint_dir + '/facenet', global_step=global_step)

                        print("Evaluating")
                        start = time.time()
                        evaluation_set = lfw.get_evaluation_batch()
<<<<<<< HEAD:train_triplet.py
                        threshold, accuracy = triplet_trian_ops.evaluate(sess, evaluation_set, embeddings,
                                                                         image_buffers_ph, is_training_ph,
                                                                         global_step_ph, global_step,
                                                                         args.batch_size, thresholds)
                        eval_summary = tf.Summary()
                        eval_summary.value.add(tag="lfw_verification_accuracy", simple_value=accuracy)
                        eval_summary.value.add(tag="lfw_verification_threshold", simple_value=threshold)
                        summary_writer.add_summary(eval_summary, global_step=global_step)
=======
                        (threshold, accuracy, precision, recall, f1) = network.evaluate(sess,
                                                                                        evaluation_set,
                                                                                        global_step,
                                                                                        batch_size=args.batch_size,
                                                                                        thresholds=thresholds)
                        elapsed = time.time() - start
                        print("Accuracy: {0:0.2f}\tThreshold: {1:0.2f}\t".format(accuracy, threshold),
                              "Precision: {0:0.2f}\tRecall: {1:0.2f}\tF-1: {2:0.2f}\t".format(precision, recall, f1),
                              "Elapsed time: {0:0.2f} secs".format(elapsed))

                        all_embeddings, image_ids = process_all_images(lfw, network, sess, global_step, args)
                        cosine_accuracy = performance.eval_cosine_sim(all_embeddings, image_ids)
                        accuracy_collection.append({"step": global_step, "accuracy": cosine_accuracy})
                        helper.to_json(accuracy_collection, os.path.join(args.checkpoint_dir, "accuracy.json"))

                        for name in ['andrew', 'erin']:
                            person_embed = all_embeddings[image_ids == lfw.name_to_idx[name], :]
                            sim = np.dot(all_embeddings, person_embed[0])
                            sorted_values = np.argsort(sim)[::-1]
                            print("Similar to {0}".format(name.title()))
                            for pls_make_functions in sorted_values[1:6]:
                                sv = sim[pls_make_functions]
                                if np.isnan(sv):
                                    raise ValueError("Comparison value is {0}. Aborting".format(sv))
                                print("\t{0} ({1:0.5f})".format(lfw.idx_to_name[image_ids[pls_make_functions]],
                                                                sv))
>>>>>>> 9f020b5fd29c1ad87572a4a0708e49516f27e374:triplet/train_triplet.py

            except KeyboardInterrupt:
                print("Keyboard Interrupt. Exiting.")
                break
        print("Saving model...")
        saver.save(sess, os.path.join(args.checkpoint_dir, 'facenet'), global_step=global_step)
        print("Saved to: {}".format(args.checkpoint_dir))
    print("Done")


def main():
    embedding_sizes = [128, 512]
    loss_funcs = ["face_net"]
    train_steps = 500000

    for es, lf in product(embedding_sizes, loss_funcs):
        checkpoint_dir = "checkpoints/triplet/" + "{}_{}".format(lf, es)
        params = ModelParams(learning_rate=0.001,
                             identities_per_batch=40,
                             train_steps=train_steps,
                             checkpoint_dir=checkpoint_dir,
                             embedding_size=es,
                             loss_func=lf,
                             pretrained_base_model=None,
                             use_center_loss=False,
                             batch_size=32)
        model_train(params)


if __name__ == "__main__":
    main()
