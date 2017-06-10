
import os
import time
import sprite
import helper
import numpy as np
import tensorflow as tf
from loss import loss_func
from functools import partial
from sample import get_triplets
from evaluation import evaluate
from tensorflow.contrib import slim
from data import Dataset, read_one_image
from networks import inception_resnet_v2
from tensorflow.contrib.tensorboard.plugins import projector

# TODO make these command line args
input_file = "fixtures/faces.json"
timestamp = helper.get_current_timestamp()
checkpoint_dir = "checkpoints/inception_resnet_v2/" + timestamp
out_tensorboard_metadata = os.path.join(checkpoint_dir, "metadata.tsv")
out_sprite_image = os.path.join(checkpoint_dir, "sprite.jpeg")

metrics_file = os.path.join(checkpoint_dir, "validation_metrics.json")
batch_size = 64
embedding_size = 128
is_training = True
learning_rate = 0.01
image_shape = (160, 160, 3)
identities_per_batch = 40
n_images_per = 25
n_validation = 3000
thresholds = np.arange(0, 4, 0.1)
sprite_thumbnail_size = [28, 28]

helper.check_dir(checkpoint_dir)

checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

print("Building graph")
graph = tf.Graph()
with graph.as_default():
    image_paths_ph = tf.placeholder(tf.string)
    read_func = partial(read_one_image,
                        is_training=is_training,
                        image_shape=image_shape)
    images = tf.map_fn(read_func, image_paths_ph, dtype=tf.float32)
    # do the network thing here
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        prelogits, endpoints = inception_resnet_v2.inception_resnet_v2(images,
                                                                       num_classes=embedding_size)
    embeddings = tf.nn.l2_normalize(prelogits, 0, name="embeddings")

    # don't run this until we've already done a batch pass of faces. We
    # compute the triplets offline, stack, and run through again
    anchors, positives, negatives = tf.unstack(tf.reshape(embeddings, [-1, 3, embedding_size]), axis=1)
    losses = loss_func(anchors, positives, negatives)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses)
    tf.summary.scalar("Loss", losses)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(checkpoint_dir,
                                           graph=graph)

print("Starting session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config).as_default() as sess:
    if checkpoint:
        global_step = int(checkpoint.split("-")[-1])
        print("Loading checkpoint: {0} Global step: {1:,}".format(checkpoint, global_step))
        saver.restore(sess, checkpoint)
    else:
        print("Initializing")
        sess.run(init_op)
        global_step = 0
    start = time.time()
    dataset = Dataset(input_file,
                      n_identities_per=identities_per_batch,
                      n_images_per=n_images_per,
                      n_eval_pairs=n_validation)

    # used to collect validation metrics
    validation_metrics = {
        "false_accept_rate": [],
        "true_accept_rate": [],
        "threshold": [],
        "global_step": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    print("Starting loop")
    while True:
        try:
            # embed and collect all current face weights
            image_paths, classes = dataset.get_train_batch()
            embeddings_np = []
            for idx in range(0, image_paths.shape[0], batch_size):
                img_path_batch = image_paths[idx: idx+batch_size]
                embed = sess.run(embeddings, feed_dict={
                    image_paths_ph: img_path_batch
                })
                embeddings_np.append(embed)
            embeddings_np = np.vstack(embeddings_np)
            triplets = get_triplets(image_paths, embeddings_np, classes)
            n_trips = triplets.shape[0]
            trip_step = batch_size - (batch_size % 3)
            for idx in range(0, n_trips, trip_step):
                feed_dict = {
                    image_paths_ph: triplets[idx: idx + trip_step]
                }
                global_step += 1
                batch_per_sec = (time.time() - start) / global_step
                if global_step % 100 == 0:
                    summary, _, loss = sess.run([merged, optimizer, losses], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step)

                else:
                    _, loss = sess.run([optimizer, losses], feed_dict=feed_dict)
                print("global step: {0:,}\tloss: {1:0.5f}\tstep/sec: {2:0.2f}".format(global_step, loss, batch_per_sec))
                if global_step % 1000 == 0:
                    saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)

                    print("Evaluating")
                    start = time.time()
                    evaluation_set = dataset.get_evaluation_batch()
                    threshold, val_rate, fa_rate, precision, recall, f1 = evaluate(sess, evaluation_set, image_paths_ph,
                                                                                   embeddings, thresholds=thresholds)
                    validation_metrics["false_accept_rate"].append(fa_rate)
                    validation_metrics["true_accept_rate"].append(val_rate)
                    validation_metrics["threshold"].append(threshold)
                    validation_metrics["global_step"].append(global_step)
                    validation_metrics["precision"].append(precision)
                    validation_metrics["recall"].append(recall)
                    validation_metrics["f1"].append(f1)

                    # keep writing to this file so we can see updates. Would be better to add to tensorboard
                    helper.to_json(validation_metrics, metrics_file)
                    elapsed = time.time() - start
                    print("VAL: {0:0.2f}\tFAR: {1:0.2f}\tThreshold: {2:0.2f}\t".format(val_rate, fa_rate, threshold),
                          "Precision: {0:0.2f}\tRecall: {1:0.2f}\tF-1: {2:0.2f}\t".format(precision, recall, f1),
                          "Elapsed time: {0:0.2f} secs".format(elapsed))

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Exiting.")
            break
    saver.save(sess, os.path.join(checkpoint_dir, 'facenet'), global_step=global_step)
    helper.to_json(validation_metrics, metrics_file)
    print("Saved to: {0}".format(checkpoint_dir))

    # projector for tensorboard
    all_image_paths, _ = dataset.get_all_files()
    projector_config = projector.ProjectorConfig()
    vis_embeddings = projector_config.embeddings.add()
    vis_embeddings.tensor_name = embeddings.name
    vis_embeddings.metadata_path = out_tensorboard_metadata
    vis_embeddings.sprite.image_path = out_sprite_image
    vis_embeddings.sprite.single_image_dim.extend(sprite_thumbnail_size)
    sprite.sprite_metadata(all_image_paths, out_tensorboard_metadata)
    sprite.make_sprite(all_image_paths, out_sprite_image, sprite_thumbnail_size)
    projector.visualize_embeddings(summary_writer, projector_config)
print("Done")
