"""
Script to export squeeze layer network to accept image buffers as input
"""


import tensorflow as tf
from tensorflow.contrib import slim
from utils.transfer import load_partial_model
from classifier.preprocess import preprocess_one_image_buffer
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import argparse


class ExportArgs(object):
    embedding_dim = 256
    model_to_load = "checkpoints/softmax_vgg/clipped_grads_2018-11-03-1914"


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to a frozen graph")
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model folder")
    parser.add_argument("-e", "--embedding", type=int, required=True, help="size of the output embeddings")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="path for output file")

    args = parser.parse_args()

    graph = tf.Graph()
    with graph.as_default():

        image_buffer_ph = tf.placeholder(tf.string, name="input_image_buffers")
        images = tf.map_fn(preprocess_one_image_buffer, image_buffer_ph, dtype=tf.float32)
        is_training = tf.constant(False, dtype=tf.bool)

        # do the network thing here
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            network_features, _ = inception_v3.inception_v3(images,
                                                            is_training=is_training,
                                                            num_classes=args.embedding)
        embeddings = tf.nn.l2_normalize(network_features, axis=1, name="l2_embedding")

    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(graph=graph, config=config) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(args.model)
        saver = load_partial_model(sess, graph, latest_checkpoint, include_scopes=['InceptionV3'])

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_node_names=['input_image_buffers', 'l2_embedding']
        )

        with tf.gfile.GFile(args.output_file, 'wb') as target:
            target.write(output_graph_def.SerializeToString())
    print("done")


if __name__ == "__main__":
    main()
