"""
Load a model, embed images, save embeddings to npz
"""

import argparse
import numpy as np
import tensorflow as tf
from export.load_graph import load_graph


def read_imagepath_file(fp):
    with open(fp) as inf:
        for line in inf.readlines():
            yield line.strip().split(",")


def load_image_buffer(fp):
    with open(fp, 'rb') as inf:
        return inf.read()


def main():
    parser = argparse.ArgumentParser(description="load a tf model and run inference on images")
    parser.add_argument("-m", "--model", help="path to frozen graph pbtext", type=str, required=True)
    parser.add_argument("-i", "--images", help="text file with paths to images", type=str, required=True)
    parser.add_argument("-o", "--output-file", help="path to output npz file", type=str, required=True)
    parser.add_argument("--input-op-name", help="name of input op, ex 'prefix/input_image_buffers:0'",
                        type=str, default="prefix/input_image_buffers:0")
    parser.add_argument("--output-op-name", help="name of the output op. ex 'prefix/l2_embedding:0'",
                        type=str, default='prefix/l2_embedding:0')

    args = parser.parse_args()
    graph = load_graph(args.model)
    input_ph = graph.get_tensor_by_name(args.input_op_name)
    embeddings_op = graph.get_tensor_by_name(args.output_op_name)
    config = tf.ConfigProto(device_count={"GPU": 0})
    class_ids = []
    embeddings = []
    with tf.Session(graph=graph, config=config) as sess:
        for cls_id, image_file in read_imagepath_file(args.images):
            image_buffer = load_image_buffer(image_file)
            embeddings.append(sess.run(embeddings_op, feed_dict={input_ph: [image_buffer]}))
            class_ids.append(cls_id)
    np.savez(args.output_file, embeddings=np.concatenate(embeddings), idx_to_name=class_ids)
    print('saved to: {}'.format(args.output_file))


if __name__ == "__main__":
    main()
