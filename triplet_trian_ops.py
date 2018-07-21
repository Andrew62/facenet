import numpy as np
from utils import helper


def inference(image_paths, sess, embeddings_op, image_buffers_ph,
              is_training_ph, global_step_ph, batch_size, global_step):
    embeddings_np = []
    for mini_idx in range(0, image_paths.shape[0], batch_size):
        mini_batch = image_paths[mini_idx: mini_idx + batch_size]
        mini_image_buffers = helper.read_buffer_vect(mini_batch)
        embeddings_np.append(sess.run(embeddings_op, feed_dict={
            image_buffers_ph: mini_image_buffers,
            is_training_ph: False,
            global_step_ph: global_step

        }))
    return np.vstack(embeddings_np)


def evaluate(sess, validation_set, embeddings_op, image_buffers_ph,
             is_training_ph, global_step_ph, global_step, batch_size, thresholds):
    col0 = validation_set[:, 0]
    col1 = validation_set[:, 1]
    all_fps = np.hstack([col0, col1])
    embeddings = inference(all_fps, sess, embeddings_op, image_buffers_ph, is_training_ph, global_step_ph,
                           batch_size, global_step)
    n_rows = validation_set.shape[0]
    col0_embeddings = embeddings[:n_rows, :]
    col1_embeddings = embeddings[n_rows:, :]
    l2_dist = l2_squared_distance(col0_embeddings, col1_embeddings, axis=1)
    true_labels = validation_set[:, -1].astype(np.int)
    threshold, acc = optimal_threshold(l2_dist, true_labels, thresholds=thresholds)
    return threshold, acc


def accuracy(pred, true_labels):
    tp = np.logical_and(pred, true_labels).sum()
    fp = np.logical_and(pred, np.logical_not(true_labels)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true_labels)).sum()
    fn = np.logical_and(np.logical_not(pred), true_labels).sum()
    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (fp + tn == 0) else fp / (fp + tn)
    acc = (tp + tn) / pred.shape[0]
    return tpr, fpr, acc


def optimal_threshold(l2_dists, true_labels, thresholds=np.arange(0.1, 4, 0.01)):
    best_accuracy = float('-inf')
    best_threshold = None
    for threshold in thresholds:
        pred = np.where(l2_dists <= threshold, 1, 0)
        _, _, acc = accuracy(pred, true_labels)
        if acc > best_accuracy:
            best_threshold = threshold
            best_accuracy = acc
    return best_threshold, best_accuracy


def l2_squared_distance(a, b, axis=None):
    return np.sum(np.power(a - b, 2), axis=axis)


def get_triplets(image_paths, embeddings, class_ids, alpha=0.2):
    unique_ids = np.unique(class_ids)
    out_fps = []

    # do this to make the comparison below to check for the same class
    for class_id in unique_ids:
        class_vectors = embeddings[class_ids == class_id, :]
        class_fps = image_paths[class_ids == class_id]
        assert class_vectors.shape[0] == class_fps.shape[0], "embedding and file name mismatch"
        out_of_class_vectors = embeddings[class_ids != class_id, :]
        out_of_class_fps = image_paths[class_ids != class_id]
        assert out_of_class_vectors.shape[0] == out_of_class_fps.shape[0], "embedding and file name mismatch"
        for anchor_idx in range(class_vectors.shape[0] - 1):
            anchor_vec = class_vectors[anchor_idx, :]
            other_positive = class_vectors[anchor_idx + 1:, :]
            pos_dist_values = l2_squared_distance(anchor_vec, other_positive, 1)
            pos_idx = np.argmax(pos_dist_values)
            positive_dist = pos_dist_values[pos_idx]
            neg_dist_values = l2_squared_distance(anchor_vec, out_of_class_vectors, 1)
            # np.where returns a list. we want the first element
            valid_negatives = np.where(neg_dist_values - positive_dist < alpha)[0]
            if len(valid_negatives) > 0:
                # input is a list of indicies
                neg_idx = np.random.choice(valid_negatives)
                out_fps.extend([class_fps[anchor_idx], class_fps[pos_idx], out_of_class_fps[neg_idx]])
    print("Generated {0:,} triplets".format(len(out_fps) // 3))
    return np.asarray(out_fps)
