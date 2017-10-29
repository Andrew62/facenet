
import numpy as np
from loss import l2_squared_distance


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


def evaluate(sess, validation_set, image_path_ph, is_training_ph, embeddings_op, global_step_ph, global_step,
             batch_size=64, thresholds=np.arange(0, 4, 0.01)):
    col0 = validation_set[:, 0]
    col1 = validation_set[:, 1]
    # TODO is it more efficient to only run unique fps then reassemble?
    all_fps = np.hstack([col0, col1])
    embeddings = []
    for idx in range(0, all_fps.shape[0], batch_size):
        batch = all_fps[idx: idx + batch_size]
        embeddings.append(sess.run(embeddings_op, feed_dict={image_path_ph: batch,
                                                             is_training_ph: False,
                                                             global_step_ph: global_step}))
    embeddings = np.vstack(embeddings)
    n_rows = validation_set.shape[0]
    col0_embeddings = embeddings[:n_rows, :]
    col1_embeddings = embeddings[n_rows:, :]
    l2_dist = l2_squared_distance(col0_embeddings, col1_embeddings, axis=1)
    true_labels = validation_set[:, -1].astype(np.int)
    threshold, acc = optimal_threshold(l2_dist, true_labels, thresholds=thresholds)
    pred = np.where(l2_dist <= threshold, 1, 0)
    precision, recall, f1 = precision_recall_f1(pred, true_labels)
    return threshold, acc, precision, recall, f1, embeddings, true_labels


if __name__ == "__main__":
    dists = np.random.random((100,))
    true_labs = np.random.randint(0, 2, (100,))
    print(optimal_threshold(dists, true_labs))
