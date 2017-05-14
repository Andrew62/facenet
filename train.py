
import helper
import numpy as np
import tensorflow as tf
from data import batch_producer
from networks import inception_v2
from tensorflow.contrib import slim
from loss import loss_func, get_negatives

input_files = ["fixtures/faces.csv"]
checkpoint_dir = "checkpoints/inception_v2/" + helper.get_current_timestamp()
summary_dir = "checkpoints/inception_v2/trian_summaries/" + helper.get_current_timestamp()
batch_size = 65
embedding_size = 128
is_training = True
learning_rate = 0.1

for d in [summary_dir, checkpoint_dir]:
    helper.check_dir(d)

checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

print("Building graph")
graph = tf.Graph()
with graph.as_default():
    anchors, positives, class_id = batch_producer(input_files, batch_size=batch_size)
    combined = tf.concat([anchors, positives], axis=0)
    # do the network thing here
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        prelogits, endpoints = inception_v2.inception_v2(combined,
                                                         num_classes=embedding_size,
                                                         is_training=is_training)
    l2 = tf.nn.l2_normalize(prelogits, dim=0)
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
    summary_writer = tf.summary.FileWriter(summary_dir,
                                           graph=graph)

print("Starting session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config).as_default() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if checkpoint:
        print("Loading checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
    else:
        print("Initializing")
        sess.run(init_op)
    global_step = 0
    try:
        while True:
            # if all the classes are the same we'll skip this step
            c_ids = sess.run(class_id)
            if np.unique(c_ids).shape[0] == 1:
                continue
            a, p = sess.run([anchor_vec, positive_vec])
            negs = get_negatives(a, p, c_ids)
            feed_dict = {
                negative_vec: negs
            }
            if global_step % 50 == 0:
                summary, _, loss = sess.run([merged, optimizer, losses], feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step)
                print("global step: {0:,}\tloss: {1:0.5f}".format(global_step, loss))
            else:
                sess.run(optimizer, feed_dict=feed_dict)

            if global_step % 1000 == 0:
                saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)
            global_step += 1
    except Exception as e:
        print(e)

    saver.save(sess, checkpoint_dir + '/facenet', global_step=global_step)
    coord.request_stop()
    coord.join(threads)
print("Done")
