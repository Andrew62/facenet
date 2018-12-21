import tensorflow as tf
from tensorflow.contrib import slim
from utils.transfer import load_partial_model
from classifier.preprocess import preprocess_one_image_buffer
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from functools import partial


def export_classifier(embedding_size, checkpoint_dir, output_file,
                      input_node_name='input_image_buffers',
                      output_node_name='l2_embedding',
                      image_shape=(160, 160, 3)):
    graph = tf.Graph()
    with graph.as_default():
        image_buffer_ph = tf.placeholder(tf.string, name=input_node_name)
        img_process = partial(preprocess_one_image_buffer, image_shape=image_shape)
        images = tf.map_fn(img_process, image_buffer_ph, dtype=tf.float32)
        is_training = tf.constant(False, dtype=tf.bool)

        # do the network thing here
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            network_features, _ = inception_v3.inception_v3(images,
                                                            is_training=is_training,
                                                            num_classes=embedding_size)
        embeddings = tf.nn.l2_normalize(network_features, axis=1, name=output_node_name)

    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(graph=graph, config=config) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest_checkpoint:
            # if there isn't a checkpoint let's try to use w/e was passed in model
            latest_checkpoint = checkpoint_dir
        print("Loading checkpoint: {}".format(latest_checkpoint))
        saver = load_partial_model(sess, graph, latest_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_node_names=[output_node_name]
        )

        with tf.gfile.GFile(output_file, 'wb') as target:
            target.write(output_graph_def.SerializeToString())
