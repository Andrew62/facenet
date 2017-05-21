
import time
import helper
import numpy as np
import tensorflow as tf
from loss import loss_func
from functools import partial
from sample import get_triplets
from networks import inception_v2
from tensorflow.contrib import slim
from data import Dataset, read_one_image

input_file = "fixtures/faces.json"
checkpoint_dir = "checkpoints/inception_v2/" + helper.get_current_timestamp()
summary_dir = "checkpoints/inception_v2/trian_summaries/" + helper.get_current_timestamp()
batch_size = 64
embedding_size = 128
is_training = True
learning_rate = 0.01
image_shape = (224, 224, 3)
identities_per_batch = 40
n_images_per = 25

for d in [summary_dir, checkpoint_dir]:
    helper.check_dir(d)

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
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        prelogits, endpoints = inception_v2.inception_v2(images,
                                                         num_classes=embedding_size,
                                                         is_training=is_training)
    embeddings = tf.nn.l2_normalize(prelogits, dim=0)
    # we'll get these offline
    anchors, positives, negatives = tf.unstack(tf.reshape(embeddings, [-1, 3]), axis=1)
    losses = loss_func(anchors, positives, negatives)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses)
    tf.summary.scalar("Loss", losses)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(slim.get_model_variables())
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summary_dir,
                                           graph=graph)

print("Starting session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config).as_default() as sess:
    if checkpoint:
        print("Loading checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
    else:
        print("Initializing")
        sess.run(init_op)
    global_step = 0
    start = time.time()
    dataset = Dataset(input_file,
                      n_identities_per=identities_per_batch,
                      n_images_per=n_images_per)
    print("Starting loop")
    while True:
        try:
            # embed and collect all current face weights
            image_paths, classes = dataset.get_batch()
            embeddings_np = []
            for idx in range(0, image_paths.shape[0], batch_size):
                img_path_batch = image_paths[idx: idx+batch_size]
                embed = sess.run(embeddings, feed_dict={
                    image_paths_ph: img_path_batch
                })
                embeddings_np.append(embed)
            embeddings_np = np.vstack(embeddings_np)
            triplets = get_triplets(image_paths, embeddings_np, classes)

            # TODO break triplets up into batches
            feed_dict = {
                image_paths_ph: triplets
            }
            global_step += 1
            epoch_per_sec = (time.time() - start) / global_step
            if global_step % 100 == 0:
                summary, _, loss = sess.run([merged, optimizer, losses], feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step)
            else:
                _, loss = sess.run([optimizer, losses], feed_dict=feed_dict)
            print("global step: {0:,}\tloss: {1:0.3f}\tstep/sec: {2:0.2f}".format(global_step, loss, epoch_per_sec))
            if global_step % 1000 == 0:
                saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)
        except KeyboardInterrupt:
            print("Keyboard Interrupt. Exiting.")
            break
    saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)
print("Done")
