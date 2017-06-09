
import numpy as np
from loss import l2_squared_distance


def validation_rate(n_accepted, n_same):
    return n_accepted / n_same


def false_accept_rate(n_false_accepted, n_diff):
    return n_false_accepted / n_diff


def optimal_threshold(l2_dists, true_labels, thresholds=np.arange(0, 4, 0.1)):
    best_validation_rate = float('-inf')
    best_false_accept_rate = float("inf")
    best_threshold = None
    for threshold in thresholds:
        cutoff = np.where(l2_dists < threshold, 1.0, 0.0)
        ta = cutoff[true_labels == 1]
        val = validation_rate(ta.sum(), true_labels.sum())
        fa = cutoff[true_labels == 0]
        n_false_accept = np.where(fa == 0, 1.0, 0.0).sum()
        n_diff = np.where(true_labels == 0, 1.0, 0.0).sum()
        far = false_accept_rate(n_false_accept, n_diff)
        if val > best_validation_rate and far < best_false_accept_rate:
            best_threshold = threshold
            best_false_accept_rate = far
            best_validation_rate = val
    return best_threshold, best_validation_rate, best_false_accept_rate


def precision_recall_f1(pred, gt):
    # intersection
    tp = np.logical_and(pred, gt).sum()
    # union
    tp_and_fp = np.logical_or(pred, gt).sum()
    precision = tp / tp_and_fp
    # false negatives
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1


def evaluate(sess, validation_set, image_path_ph, embeddings_op, batch_size=64, thresholds=np.arange(0, 4, 0.1)):
    col0 = validation_set[:, 0]
    col1 = validation_set[:, 1]
    # TODO is it more efficient to only run unique fps then reassemble?
    all_fps = np.hstack([col0, col1])
    embeddings = []
    for idx in range(0, all_fps.shape[0], batch_size):
        batch = all_fps[idx: idx + batch_size]
        embeddings.append(sess.run(embeddings_op, feed_dict={image_path_ph: batch}))
    embeddings = np.vstack(embeddings)
    n_rows = validation_set.shape[0]
    col0_embeddings = embeddings[:n_rows, :]
    col1_embeddings = embeddings[n_rows:, :]
    l2_dist = l2_squared_distance(col0_embeddings, col1_embeddings, axis=1)
    true_labels = validation_set[:, -1].astype(np.int)
    threshold, val_rate, fa_rate = optimal_threshold(l2_dist, true_labels, thresholds=thresholds)
    pred = np.where(l2_dist < threshold, 1, 0)
    precision, recall, f1 = precision_recall_f1(pred, true_labels)
    return threshold, val_rate, fa_rate, precision, recall, f1
