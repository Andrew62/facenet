import os
import numpy as np
import tensorflow as tf
from train_classifier import read_train_csv
from utils import helper


checkpoint_dir = 'checkpoints/softmax/2018-07-29-1316'
input_csv_path = "fixtures/faces/representative-faces/face_centers.csv"


class OutArgs:
    pass


out_args = OutArgs()
out_args.batch_size = 64
out_args.checkpoint_dir = checkpoint_dir

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print("Loading checkpoint " + latest_checkpoint)
    saver = tf.train.import_meta_graph(latest_checkpoint + ".meta")
    saver.restore(sess, latest_checkpoint)
    graph = tf.get_default_graph()
    input_ph = graph.get_tensor_by_name("input_image_buffers:0")
    embeddings_op = graph.get_tensor_by_name("l2_embedding:0")
    is_training_ph = graph.get_tensor_by_name("is_training:0")

    data = read_train_csv(input_csv_path)
    print("Processing")
    embeddings_list = []
    class_codes_list = []
    for idx in range(0, data.shape[0], out_args.batch_size):
        batch = data[idx: idx + out_args.batch_size]
        buffers = helper.read_buffer_vect(batch[:, 1])
        embeddings_list.append(sess.run(embeddings_op, feed_dict={input_ph: buffers, is_training_ph: False}))
    embeddings = np.concatenate(embeddings_list)
    np.savez(os.path.join(checkpoint_dir, "representative-embeddings.npz"), embeddings=embeddings_list, class_codes=data[:, 0])
    print(embeddings.shape)

print("done")