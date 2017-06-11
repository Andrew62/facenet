
import sys
import time
import numpy as np
import tensorflow as tf
from functools import partial
from argparse import ArgumentParser
from tensorflow.contrib import slim
from networks import inception_resnet_v2
from data import read_one_image, Dataset


def main():
    parser = ArgumentParser(description="Program to run inference with a facenet model")
    parser.add_argument("-i", "--input_json", help="path to input json file",
                        type=str, required=True)
    parser.add_argument("-o", "--out_npz", help="output embeddings as a npz file",
                        type=str, required=True)
    parser.add_argument("-c", "--checkpoint_dir", help="model checkpoint directory to use",
                        type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="number of images per batch",
                        type=int, default=64)
    args = parser.parse_args()

    embedding_size = 128
    is_training = False
    image_shape = (160, 160, 3)

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
                                                                           num_classes=embedding_size,
                                                                           dropout_keep_prob=1.0)
        l2 = tf.nn.l2_normalize(prelogits, 0)
        saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session(graph=graph).as_default() as sess:
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if checkpoint:
            print("Loading checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print("No checkpoint found! Exiting")
            sys.exit(0)
        embeddings = []
        dataset = Dataset(args.input_json)
        file_paths, class_codes = dataset.get_all_files()
        counter = 1
        start = time.time()
        for idx in range(0, len(file_paths), args.batch_size):
            embedding = sess.run(l2, feed_dict={
                image_paths_ph: file_paths[idx: idx+args.batch_size]
            })
            embeddings.append(embedding)
            batch_per_sec = (time.time() - start) / counter
            print("processed batch {0:,}\tbatch/sec: {1:0.2f}".format(counter, batch_per_sec))
            counter += 1

        embeddings = np.vstack(embeddings)
        class_codes = np.vstack(class_codes)
        np.savez(args.out_npz, embeddings=embeddings, class_codes=class_codes)

if __name__ == "__main__":
    main()
