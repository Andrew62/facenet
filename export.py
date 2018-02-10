import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from networks import inception_resnet_v2
from transfer import load_partial_model
from argparse import ArgumentParser
from data import Dataset


def read_one_buffer(buffer, **kwargs):
    img_shape = kwargs.pop("image_shape", (299, 299, 3))
    # decode buffer as an image
    image = tf.image.decode_image(buffer, channels=img_shape[-1])

    image = tf.image.resize_image_with_crop_or_pad(image, img_shape[0], img_shape[1])
    image.set_shape(img_shape)
    return tf.image.per_image_standardization(image)


def load_buffer(fp):
    with open(fp, 'rb') as inf:
        return inf.read()


def main():

    parser = ArgumentParser(description="export a facenet model that takes image buffers as input")
    parser.add_argument("-c", "--checkpoint_name", help="checkpoint name to load. Should prob not have an extension",
                        type=str, required=True)
    parser.add_argument("-o", "--out", help="location of the output checkpoint file",
                        type=str)
    parser.add_argument("-i", "--input_faces", help="input faces json file",
                        default="fixtures/faces.json")
    parser.add_argument("-b", "--batch_size", help="batch size to use when generating embeddings", default=64,
                        type=int)

    args = parser.parse_args()

    graph = tf.Graph()
    with graph.as_default():
        input_buffers = tf.placeholder(tf.string, name="input_image_buffers")
        image_batch = tf.map_fn(read_one_buffer, input_buffers, dtype=tf.float32)
        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, endpoints = inception_resnet_v2.inception_resnet_v2(image_batch,
                                                                        is_training=tf.constant(False, dtype=tf.bool),
                                                                        num_classes=2048)
        with tf.variable_scope("face_embedding"):
            weights_1 = tf.get_variable("weights_1", shape=(2048, 2048),
                                        initializer=tf.contrib.layers.xavier_initializer())
            layer_1 = tf.nn.relu(tf.matmul(logits, weights_1))

            layer_1_dropout = tf.cond(tf.constant(False, dtype=tf.bool),  # keeping this conditional here to match train
                                      true_fn=lambda: tf.nn.dropout(layer_1, keep_prob=0.8),
                                      false_fn=lambda: layer_1)
            weights_2 = tf.get_variable("weights_2", shape=(2048, 200),
                                        initializer=tf.contrib.layers.xavier_initializer())
            layer_2 = tf.matmul(layer_1_dropout, weights_2)

        embeddings = tf.nn.l2_normalize(layer_2, 1, 1e-10, name="l2_embedding")

        # use these so we can grab them easier later. Don't like using "get_tensor_by_name" or w/e
        tf.add_to_collection("input_buffers", input_buffers)
        tf.add_to_collection("out_embeddings", embeddings)

    config = tf.ConfigProto(device_count={"GPU":1})
    with tf.Session(graph=graph, config=config) as sess:
        dataset = Dataset(args.input_faces)
        saver = load_partial_model(sess,
                                   graph,
                                   # forever excluding these scopes b/c of transfer learning
                                   ["InceptionResnetV2/AuxLogits", "RMSProp"],
                                   args.checkpoint_name)
        all_images, image_ids = dataset.get_all_files()
        all_embeddings = []
        for idx in range(0, len(all_images), args.batch_size):
            print("{0:0.1%}".format(idx  / len(all_images)))
            batch = list(map(load_buffer, all_images[idx:idx+args.batch_size]))
            all_embeddings.append(sess.run(embeddings, feed_dict={
                input_buffers: batch
            }))
        all_embeddings = np.vstack(all_embeddings)        
        np.savez(args.out + "_embeddings.npz",
                 embeddings=all_embeddings,
                 class_codes=np.array(image_ids))
        
        saver.save(sess, args.out)
    print("exported to: {}".format(args.out))
    print("collection input_buffers contains input placeholder")
    print("collection out_embeddings contains embedding result")


if __name__ == "__main__":
    main()
