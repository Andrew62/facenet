
import sys
import numpy as np
import tensorflow as tf
from facenet import FaceNet
from functools import partial
from argparse import ArgumentParser
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
    learning_rate = 0.01

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
                         true_fn=lambda: tf.map_fn(read_one_train, image_paths_ph, dtype=tf.float32),
                         false_fn=lambda: tf.map_fn(read_one_test, image_paths_ph, dtype=tf.float32))
        network = FaceNet(images,
                          is_training_ph,
                          embedding_size,
                          global_step_ph,
                          learning_rate)
    with tf.Session(graph=graph).as_default() as sess:
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        saver = tf.train.Saver()
        if checkpoint:
            print("Loading checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print("No checkpoint found! Exiting")
            sys.exit(0)
        dataset = Dataset(args.input_json)
        file_paths, class_codes = dataset.get_all_files()
        out_embeddings = network.inference(sess,
                                           file_paths,
                                           args.batch_size,
                                           False,
                                           image_paths_ph,
                                           is_training_ph,
                                           1,
                                           global_step_ph)
        class_codes = np.vstack(class_codes)
        np.savez(args.out_npz, embeddings=out_embeddings, class_codes=class_codes)
    print("Done")


if __name__ == "__main__":
    main()
