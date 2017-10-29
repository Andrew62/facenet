
import numpy as np


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


if __name__ == "__main__":
    dists = np.random.random((100,))
    true_labs = np.random.randint(0, 2, (100,))
    print(optimal_threshold(dists, true_labs))
