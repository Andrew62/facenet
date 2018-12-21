import os
import tempfile
import numpy as np
import tensorflow as tf
from triplet.data import Dataset
from triplet.performance import optimal_threshold
from export.load_graph import load_graph
from itertools import combinations
from export.export_classifier import export_classifier
from multiprocessing import Process
from utils.helper import get_current_timestamp


def load_img(fp):
    with open(fp, 'rb') as inf:
        return inf.read()
    
load_img_np = np.vectorize(load_img)


def l2_squared_distance(a, b, axis=None):
    return np.sum(np.power(a - b, 2), axis=axis)


def inference(image_paths, sess, embeddings_op, image_buffer_ph, batch_size):
    embeddings_np = []
    for mini_idx in range(0, image_paths.shape[0], batch_size):
        mini_batch = load_img_np(image_paths[mini_idx: mini_idx + batch_size])
        embeddings_np.append(sess.run(embeddings_op, feed_dict={
            image_buffer_ph: mini_batch,
        }))
    return np.vstack(embeddings_np)


def gen_pairs(embeddings, class_ids):
    """
    :param embeddings: image embeddings
    :param class_ids:  true labels for each emebedding
    :return: col1, col2, is_same
    """
    embed1 = []
    embed2 = []
    is_same = []
    unique_ids = np.unique(class_ids)
    for class_id in unique_ids:
        class_vectors = embeddings[class_ids == class_id, :]
        out_of_class_vectors = embeddings[class_ids != class_id, :]

        out_of_class_idxs = list(range(out_of_class_vectors.shape[0]))
        in_class_idxs = list(range(class_vectors.shape[0]))

        positive_count = 0
        for pos1_idx, pos2_idx in combinations(in_class_idxs, 2):
            embed1.append(class_vectors[pos1_idx, :])
            embed2.append(class_vectors[pos2_idx, :])
            is_same.append(1)
            positive_count += 1

        for _ in range(positive_count):
            pos_idx = np.random.choice(in_class_idxs)
            neg_idx = np.random.choice(out_of_class_idxs)
            embed1.append(class_vectors[pos_idx, :])
            embed2.append(out_of_class_vectors[neg_idx, :])
            is_same.append(0)
    embed1 = np.vstack(embed1)
    embed2 = np.vstack(embed2)
    is_same = np.array(is_same)
    return embed1, embed2, is_same


def check_tensor_name(tensor_name):
    """
    Tensors should specify a device using <tensor name>:<device number>
    """
    if ":" not in tensor_name:
        tensor_name = tensor_name + ":0"
    return tensor_name


def poor_mans_log(msg):
    tmp_dir = tempfile.gettempdir()
    out_error_log_file = os.path.join(tmp_dir, "lfw_eval_{}.log".format(get_current_timestamp()))
    with open(out_error_log_file, 'w') as target:
        target.write(msg)


def lfw_eval(lfw_fp, checkpoint_dir, input_tensor_name, embedding_name, batch_size, embedding_size, global_step,
             image_shape):
    try:
        input_tensor_name = check_tensor_name(input_tensor_name)
        embedding_name = check_tensor_name(embedding_name)
        tmp_model = tempfile.NamedTemporaryFile(delete=False)
        export_classifier(embedding_size, checkpoint_dir, tmp_model.name, image_shape)
        graph = load_graph(tmp_model.name)
        image_buffer_ph = graph.get_tensor_by_name(input_tensor_name)
        embeddings_op = graph.get_tensor_by_name(embedding_name)
        lfw = Dataset(lfw_fp)
        file_paths, class_ids = lfw.get_all_files()
        class_ids = np.array(class_ids)
        file_paths = np.array(file_paths)
        thresholds = np.arange(0.1, 4, 0.01)

        config = tf.ConfigProto(device_count={"GPU": 0})
        with tf.Session(graph=graph, config=config) as sess:
            summary_writer = tf.summary.FileWriter(checkpoint_dir)
            lfw_embeddings = inference(file_paths, sess, embeddings_op, image_buffer_ph, batch_size)
            embed1, embed2, is_same = gen_pairs(lfw_embeddings, class_ids)
            l2_dist = l2_squared_distance(embed1, embed2, axis=1)
            threshold, accuracy = optimal_threshold(l2_dist, is_same, thresholds=thresholds)
            eval_summary = tf.Summary()
            eval_summary.value.add(tag="lfw_verification_accuracy", simple_value=accuracy)
            eval_summary.value.add(tag="lfw_verification_threshold", simple_value=threshold)
            summary_writer.add_summary(eval_summary, global_step=global_step)
        msg = "accuracy: {}\tthreshold: {}\n".format(accuracy, threshold)
    except Exception as e:
        msg = str(e)
        accuracy, threshold = -1., -1.
    poor_mans_log(msg)
    return accuracy, threshold


class EvalProcess(Process):
    def __init__(self,
                 lfw_fp,
                 checkpoint_dir,
                 input_tensor_name,
                 embedding_name,
                 batch_size,
                 embedding_size,
                 global_step,
                 **kwargs):
        super().__init__(**kwargs)
        self.lfw_fp = lfw_fp
        self.checkpoint_dir = checkpoint_dir
        self.input_tensor_name = check_tensor_name(input_tensor_name)

        self.embedding_name = check_tensor_name(embedding_name)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.global_step = global_step

    def run(self):
        return lfw_eval(self.lfw_fp, self.checkpoint_dir, self.input_tensor_name, self.embedding_name,
                        self.batch_size, self.embedding_size, self.global_step)


def lfw_eval_async(lfw_fp, checkpoint_dir, input_tensor_name, embedding_name, batch_size, embedding_size, global_step):
    p = EvalProcess(lfw_fp, checkpoint_dir, input_tensor_name, embedding_name, batch_size, embedding_size, global_step)
    p.start()
    p.join()
