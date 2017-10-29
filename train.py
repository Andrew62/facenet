
import os
import time
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

# TODO make these command line args
input_file = "fixtures/faces.json"
timestamp = helper.get_current_timestamp()
checkpoint_dir = "checkpoints/inception_resnet_v2/" + timestamp
metrics_file = os.path.join(checkpoint_dir, "validation_metrics.json")
batch_size = 64
embedding_size = 128
is_training = True
starting_learning_rate = 0.01
image_shape = (160, 160, 3)
identities_per_batch = 100
n_images_per = 25
n_validation = 3000
thresholds = np.arange(0, 4, 0.1)
embeddings_npz = "embeddings/embeddings.npz"
checkpoint_exclude_scopes = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits", "RMSProp"]

os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
inception_resnet_v2_pretrained = "checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt"

print("Building graph")
graph = tf.Graph()
with graph.as_default():
    image_paths_ph = tf.placeholder(tf.string)
    global_step_ph = tf.placeholder(tf.int32)
    is_training_ph = tf.placeholder(tf.bool)
    # do this so we can change behavior
    read_one_train = partial(read_one_image,
                             is_training=True,
                             image_shape=image_shape)
    read_one_test = partial(read_one_image,
                            is_training=False,
                            image_shape=image_shape)
    images = tf.cond(is_training_ph,
                     lambda: tf.map_fn(read_one_train, image_paths_ph, dtype=tf.float32),
                     lambda: tf.map_fn(read_one_train, image_paths_ph, dtype=tf.float32))

    # do the network thing here
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        prelogits, endpoints = inception_resnet_v2.inception_resnet_v2(images,
                                                                       is_training=is_training,
                                                                       num_classes=embedding_size)
    embeddings = tf.nn.l2_normalize(prelogits, 0, name="embeddings")

    # don't run this until we've already done a batch pass of faces. We
    # compute the triplets offline, stack, and run through again
    anchors, positives, negatives = tf.unstack(tf.reshape(embeddings, [-1, 3, embedding_size]), axis=1)
    losses = loss_func(anchors, positives, negatives)
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([losses] + regularization_loss, name="total_loss")
    tf.summary.scalar("Triplet_Loss", losses)
    tf.summary.scalar("Total_Loss", total_loss)
    learning_rate = tf.train.exponential_decay(starting_learning_rate, decay_rate=0.96,
                                               decay_steps=10000, staircase=True,
                                               global_step=global_step_ph)
    tf.summary.scalar("Learning_Rate", learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)
    merged = tf.summary.merge_all()

print("Starting session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config).as_default() as sess:
    summary_writer = tf.summary.FileWriter(checkpoint_dir,
                                           graph=graph)
    to_restore = {}
    to_init = []
    variables = graph.get_collection("variables")
    for variable in variables:
        v_name = variable.name.split(":")[0]
        if not any(map(lambda x: x in v_name, checkpoint_exclude_scopes)):
            to_restore[v_name] = variable
        else:
            to_init.append(variable)
    saver = tf.train.Saver(var_list=variables)
    model_restore = tf.train.Saver(var_list=to_restore)
    model_restore.restore(sess, inception_resnet_v2_pretrained)
    sess.run(tf.variables_initializer(var_list=to_init))
    global_step = 0
    start = time.time()
    dataset = Dataset(input_file,
                      n_identities_per=identities_per_batch,
                      n_images_per=n_images_per,
                      n_eval_pairs=n_validation)

    # used to collect validation metrics
    validation_metrics = {
        "accuracy": [],
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
                    image_paths_ph: img_path_batch,
                    global_step_ph: global_step,
                    is_training_ph: True
                })
                embeddings_np.append(embed)
            embeddings_np = np.vstack(embeddings_np)
            triplets = get_triplets(image_paths, embeddings_np, classes)
            n_trips = triplets.shape[0]
            trip_step = batch_size - (batch_size % 3)
            for idx in range(0, n_trips, trip_step):
                feed_dict = {
                    image_paths_ph: triplets[idx: idx + trip_step],
                    is_training_ph: True,
                    global_step_ph: global_step
                }
                global_step += 1
                batch_per_sec = (time.time() - start) / global_step
                if global_step % 100 == 0:
                    summary, _, loss = sess.run([merged, optimizer, total_loss], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step)
                else:
                    _, loss = sess.run([optimizer, total_loss], feed_dict=feed_dict)
                print("global step: {0:,}\tloss: {1:0.5f}\tstep/sec: {2:0.2f}".format(global_step,
                                                                                      loss,
                                                                                      batch_per_sec))
                if global_step % 1000 == 0:
                    saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)

                    print("Evaluating")
                    start = time.time()
                    evaluation_set = dataset.get_evaluation_batch()
                    (threshold, accuracy, precision, recall,
                     f1, eval_embeddings, class_codes) = evaluate(sess, evaluation_set, image_paths_ph,
                                                                  is_training_ph, embeddings, global_step_ph,
                                                                  global_step, thresholds=thresholds)
                    validation_metrics["accuracy"].append(accuracy)
                    validation_metrics["threshold"].append(threshold)
                    validation_metrics["global_step"].append(global_step)
                    validation_metrics["precision"].append(precision)
                    validation_metrics["recall"].append(recall)
                    validation_metrics["f1"].append(f1)

                    # keep writing to this file so we can see updates. Would be better to add to tensorboard
                    helper.to_json(validation_metrics, metrics_file)
                    np.savez(embeddings_npz, embeddings=eval_embeddings, class_codes=class_codes)
                    elapsed = time.time() - start
                    print("Accuracy: {0:0.2f}\tThreshold: {1:0.2f}\t".format(accuracy, threshold),
                          "Precision: {0:0.2f}\tRecall: {1:0.2f}\tF-1: {2:0.2f}\t".format(precision, recall, f1),
                          "Elapsed time: {0:0.2f} secs".format(elapsed))
                    all_images, image_ids = dataset.get_all_files()
                    all_images_np, image_ids_np = np.array(all_images), np.array(image_ids)
                    all_embeddings = np.zeros((len(all_images), embedding_size), dtype=np.float32)
                    for batch_idx in range(0, len(all_images), batch_size):
                        all_embeddings[batch_idx:batch_idx + batch_size] = sess.run(embeddings, feed_dict={
                            image_paths_ph: all_images_np[batch_idx: batch_idx + batch_size],
                            is_training_ph: False,
                            global_step_ph: global_step
                        })
                    for name in ['andrew', 'erin']:
                        person_embed = all_embeddings[image_ids_np == dataset.name_to_idx[name], :]
                        sim = np.dot(all_embeddings, person_embed[0])
                        sorted_values = np.argsort(sim)[::-1]
                        print("Similar to {0}".format(name.title()))
                        for pls_make_functions in sorted_values[1:6]:
                            print("\t{0} ({1:0.5f})".format(dataset.idx_to_name[image_ids_np[pls_make_functions]],
                                                            sim[pls_make_functions]))

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Exiting.")
            break
    saver.save(sess, os.path.join(checkpoint_dir, 'facenet'), global_step=global_step)
    helper.to_json(validation_metrics, metrics_file)
    print("Saved to: {0}".format(checkpoint_dir))
print("Done")
