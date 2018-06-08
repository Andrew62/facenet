import tensorflow as tf
from train_classifier import read_train_csv, process_all_images


checkpoint_dir = 'checkpoints/softmax/2018-05-20-1418'
input_csv_path = "fixtures/face-detect-site.csv"


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
    embed, img_id = process_all_images(sess, "final", data, input_ph, embeddings_op,
                                       out_args, is_training_ph)
    print(embed.shape, img_id.shape)

print("done")