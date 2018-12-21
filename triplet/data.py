
import random
import numpy as np
from utils import helper


class Dataset(object):
    def __init__(self, class_fp, **kwargs):
        self.class_data = helper.load_json(class_fp)
        self.class_idxs = list(self.class_data.keys())
        self.n_identities_per = min(kwargs.pop("n_identities_per", 40), len(self.class_data.keys()))
        self.n_images_per = kwargs.pop("n_images_per", 25)
        self.n_eval_pairs = kwargs.pop("n_eval_pairs", 10000)

        # use these to cache our validation pairs
        self.eval_fps = []

    @property
    def total_images(self):
        return sum(map(len, self.class_data.values()))

    @property
    def n_classes(self):
        return len(self.class_idxs)

    def get_train_batch(self, **kwargs):
        n_identities_per = kwargs.pop("n_identities_per", self.n_identities_per)
        n_images_per = kwargs.pop("n_images_per", self.n_images_per)

        out_fps = []
        out_ids = []
        random.shuffle(self.class_idxs)
        for cidx in self.class_idxs[:n_identities_per]:
            fps = self.class_data[cidx]
            n_images = min(len(fps), n_images_per)
            random.shuffle(fps)
            for fp in fps[:n_images]:
                out_fps.append(fp)
                out_ids.append(cidx)
        return np.asarray(out_fps), np.asarray(out_ids)

    def get_all_files(self):
        """
        :return: [file paths], [class ids]
        """
        out_fps = []
        out_ids = []
        for cidx, file_paths in self.class_data.items():
            for fp in file_paths:
                out_fps.append(fp)
                out_ids.append(cidx)
        return out_fps, out_ids

