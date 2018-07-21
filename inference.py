import os
import sys
import json
import numpy as np
import tensorflow as tf
from triplet_trian_ops import FaceNet
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Program to run inference with a facenet model")
    parser.add_argument("-i", "--inputs", help="path to input image file",
                        type=str, required=True, nargs="*")
    # parser.add_argument("-o", "--out_npz", help="output embeddings as a npz file",
    #                     type=str, required=True)
    parser.add_argument("-c", "--checkpoint_dir", help="model checkpoint directory to use",
                        type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="number of images per batch",
                        type=int, default=64)
    args = parser.parse_args()

    file_embeddings = os.path.join(args.checkpoint_dir, "embeddings.npz")
    file_idx_to_name = os.path.join(args.checkpoint_dir, "idx_to_name.json")
    data = np.load(file_embeddings)
    embeddings = np.float128(data['embeddings'])
    classes = data['class_codes']
    with open(file_idx_to_name) as inf:
        idx_to_name = json.load(inf)

    embedding_size = 128
    image_shape = (160, 160, 3)
    learning_rate = 0.01

    print("Building graph")
    graph = tf.Graph()
    with graph.as_default():
        image_paths_ph = tf.placeholder(tf.string)
        global_step_ph = tf.placeholder(tf.int32)
        is_training_ph = tf.placeholder(tf.bool)
        network = FaceNet(image_paths_ph,
                          is_training_ph,
                          embedding_size,
                          global_step_ph,
                          learning_rate,
                          image_shape)
    with tf.Session(graph=graph).as_default() as sess:
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        saver = tf.train.Saver(graph.get_collection("variables"))
        if checkpoint:
            print("Loading checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print("No checkpoint found! Exiting")
            sys.exit(0)
        out_embeddings = network.inference(sess,
                                           np.array(args.inputs),
                                           args.batch_size,
                                           False, 1)
        cosine_sim = np.dot(embeddings, np.float128(out_embeddings.T))
        print(cosine_sim.shape)
        for idx, input_f in enumerate(args.inputs):
            print("Input: {0}".format(input_f))
            for row_idx in np.argsort(cosine_sim[:, idx])[:-3:-1]:
                print('\t', idx_to_name[str(classes[row_idx])], cosine_sim[row_idx, idx])


if __name__ == "__main__":
    main()
