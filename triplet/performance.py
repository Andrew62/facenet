
import numpy as np


def accuracy(pred, true_labels):
    correct = (pred == true_labels).sum()
    return correct / pred.shape[0]


def optimal_threshold(l2_dists, true_labels, thresholds=np.arange(0.1, 4, 0.01)):
    best_accuracy = float('-inf')
    best_threshold = None
    for threshold in thresholds:
        pred = np.where(l2_dists <= threshold, 1, 0)
        acc = accuracy(pred, true_labels)
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


def eval_cosine_sim(all_embeddings, image_ids, n_samples=10):
    n_items = all_embeddings.shape[0]
    ref_size = np.int(np.ceil(n_items * 0.8))
    samples = []
    for _ in range(n_samples):
        reference_mask = np.random.choice(n_items, ref_size, replace=False)
        # since starting with ones, everything is true
        eval_mask = np.ones(n_items, dtype=np.bool)
        eval_mask[reference_mask] = False
        reference_embeddings = all_embeddings[reference_mask]
        reference_ids = image_ids[reference_mask]
        test_embeddings = all_embeddings[eval_mask]
        test_ids = image_ids[eval_mask]
        pred_cosine = np.dot(reference_embeddings, test_embeddings.T)
        pred_idx = np.argmax(pred_cosine, axis=0)
        pred_ids = reference_ids[pred_idx]
        samples.append(np.sum(test_ids == pred_ids)/test_ids.shape[0])
    return np.mean(samples)
