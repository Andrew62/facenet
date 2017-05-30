import numpy as np
from loss import l2_squared_distance


def get_triplets(image_paths, embeddings, class_ids):
    unique_ids = np.unique(class_ids)
    out_fps = []

    # do this to make the comparison below to check for the same class
    for idx, class_id in enumerate(unique_ids):
        class_vectors = embeddings[class_ids == class_id]
        class_fps = image_paths[class_ids == class_id]
        out_of_class_vectors = embeddings[class_ids != class_id]
        out_of_class_fps = image_paths[class_ids != class_id]
        best_neg_dist = float('inf')
        best_pos_dist = float('-inf')
        best_anchor = None
        best_pos = None
        best_neg = None
        for anchor_idx in range(class_vectors.shape[0]):
            pos_dist = l2_squared_distance(class_vectors[anchor_idx], class_vectors)
            pos_idx = np.argmax(pos_dist)
            neg_dist = l2_squared_distance(class_vectors[anchor_idx], out_of_class_vectors)
            neg_idx = np.argmin(neg_dist)
            if neg_dist < best_neg_dist and pos_dist > best_pos_dist:
                best_neg_dist = neg_dist
                best_pos_dist = pos_dist
                best_pos = pos_idx
                best_neg = neg_idx
                best_anchor = anchor_idx
        if best_neg is not None and best_neg is not None:
            out_fps.extend([class_fps[best_anchor], class_fps[best_pos], out_of_class_fps[best_neg]])

    return np.asarray(out_fps)
