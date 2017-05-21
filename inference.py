import sys
import numpy as np
import tensorflow as tf
from functools import partial
from networks import inception_v2
from tensorflow.contrib import slim
from data import read_one_image, Dataset

input_file = "fixtures/faces.json"
checkpoint_dir = "checkpoints/inception_v2/2017-05-21-1221"

embedding_size = 128
is_training = False
batch_size = 4
image_shape = (224, 224, 3)

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
    l2 = tf.nn.l2_normalize(prelogits, dim=0)
    saver = tf.train.Saver(slim.get_model_variables(),
                           keep_checkpoint_every_n_hours=1)
    init_op = tf.local_variables_initializer()

with tf.Session(graph=graph).as_default() as sess:
    if checkpoint:
        print("Loading checkpoint: {}".format(checkpoint))
        sess.run(init_op)
        saver.restore(sess, checkpoint)
    else:
        print("No checkpoint found! Exiting")
        sys.exit(0)
    embeddings = []
    class_codes = []
    dataset = Dataset(input_file)
    file_paths, class_codes = dataset.get_all_files()
    counter = 1
    for idx in range(0, len(file_paths), batch_size):
        print("Processing batch {:,}".format(counter))
        embedding = sess.run(l2, feed_dict={
            image_paths_ph: file_paths[idx: idx+batch_size]
        })
        embeddings.append(embedding)
        counter += 1

    embeddings = np.vstack(embeddings)
    class_codes = np.vstack(class_codes)
    np.savez("embeddings/embeddings.npz", embeddings=embeddings, class_codes=class_codes)
