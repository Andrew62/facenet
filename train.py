
import numpy as np
import tensorflow as tf
from data import batch_producer
from networks import inception_v2
from tensorflow.contrib import slim
from loss import loss_func, get_negatives

input_files = ["fixtures/faces.csv"]
out_dir = "checkpoints/inception_v2"
batch_size = 40
embedding_size = 128
is_training = True
training_steps = int(5e4)
learning_rate = 1e-3

graph = tf.Graph()
with graph.as_default():
    anchors, positives, class_id = batch_producer(input_files, batch_size=batch_size)
    combined = tf.concat([anchors, positives], axis=0)
    # do the network thing here
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        prelogits, endpoints = inception_v2.inception_v2(combined,
                                                         num_classes=embedding_size,
                                                         is_training=is_training)
    l2 = tf.nn.l2_normalize(prelogits, 0)
    anchor_vec = tf.slice(l2, [0, 0], [batch_size, -1])
    positive_vec = tf.slice(l2, [batch_size, 0], [-1, -1])
    # we'll get these offline
    negative_vec = tf.placeholder(tf.float32, [batch_size, embedding_size])
    losses = loss_func(anchor_vec, positive_vec, negative_vec)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses)
    tf.summary.scalar("Loss", losses)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(slim.get_model_variables(),
                           keep_checkpoint_every_n_hours=1)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(out_dir + "/train_summary", graph=graph)


with tf.Session(graph=graph).as_default() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init_op)
    global_step = 0
    skips = 0
    while global_step < training_steps:
        # if all the classes are the same we'll skip this step
        c_ids = sess.run(class_id)
        if np.unique(c_ids).shape[0] == 1:
            continue
        a, p = sess.run([anchor_vec, positive_vec])
        negs = get_negatives(a, p, c_ids)
        feed_dict = {
            negative_vec: negs
        }
        if global_step % 10 == 0:
            summary, _, loss = sess.run([merged, optimizer, losses], feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step)
            print("global step: {0:,}\tloss: {1:0.5f}".format(global_step, loss))
        else:
            sess.run(optimizer, feed_dict=feed_dict)

        if (global_step + 1) % 1000 == 0:
            saver.save(sess, out_dir + "/facenet", global_step=global_step)
        global_step += 1

    saver.save(sess, out_dir + "/facenet", global_step=global_step)
    coord.request_stop()
    coord.join(threads)
