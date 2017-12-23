
import tensorflow as tf
from tensorflow.contrib import slim
from networks import inception_resnet_v2
from transfer import load_partial_model
from argparse import ArgumentParser


def read_one_buffer(buffer, **kwargs):
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    # decode buffer as an image
    image = tf.image.decode_image(buffer, channels=img_shape[-1])

    image = tf.image.resize_image_with_crop_or_pad(image, img_shape[0], img_shape[1])
    image.set_shape(img_shape)
    return tf.image.per_image_standardization(image)

def main():

    parser = ArgumentParser(description="export a facenet model that takes image buffers as input")
    parser.add_argument("-c", "--checkpoint_name", help="checkpoint name to load. Should prob not have an extension",
                        type=str)
    parser.add_argument("-o", "--out", help="location of the output checkpoint file",
                        type=str)

    args = parser.parse_args()

    graph = tf.Graph()
    with graph.as_default():
        input_buffers = tf.placeholder(tf.string, name="input_image_buffers")
        image_batch = tf.map_fn(read_one_buffer, input_buffers, dtype=tf.float32)
        # do the network thing here
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            prelogits, endpoints = inception_resnet_v2.inception_resnet_v2(image_batch,
                                                                           is_training=tf.constant(False, dtype=tf.bool),
                                                                           num_classes=128)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name="l2_embedding")

        # use these so we can grab them easier later. Don't like using "get_tensor_by_name" or w/e
        tf.add_to_collection("input_buffers", input_buffers)
        tf.add_to_collection("out_embeddings", embeddings)

    with tf.Session(graph=graph) as sess:
        saver = load_partial_model(sess,
                                   graph,
                                   # forever excluding these scopes b/c of transfer learning
                                   ["InceptionResnetV2/AuxLogits", "RMSProp"],
                                   args.checkpoint_file)
        saver.save(sess, args.out)
    print("exported to: {}".format(args.out))
    print("collection input_buffers contains input placeholder")
    print("collection out_embeddings contains embedding result")


if __name__ == "__main__":
    main()
