import numpy as np
from loss import l2_squared_distance


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
            neg_dist_values = l2_squared_distance(class_vectors[anchor_idx, :], out_of_class_vectors, axis=1)
            # np.where returns a list. we want the first element
            valid_negatives = np.where(neg_dist_values - positive_dist < alpha)[0]
            if len(valid_negatives) > 0:
                # input is a list of indicies
                neg_idx = np.random.choice(valid_negatives)
                out_fps.extend([class_fps[anchor_idx], class_fps[pos_idx], out_of_class_fps[neg_idx]])
    print("Generated {0:,} triplets".format(len(out_fps)//3))
    return np.asarray(out_fps)
