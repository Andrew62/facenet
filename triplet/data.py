
import random
import numpy as np
from utils import helper
from itertools import combinations


class Dataset(object):
    def __init__(self, class_fp, **kwargs):
        self.class_data = helper.load_json(class_fp)
        self.idx_to_name = dict((idx, name) for idx, name in enumerate(sorted(self.class_data.keys())))
        self.name_to_idx = dict((name, idx) for idx, name in self.idx_to_name.items())
        self.n_identities_per = min(kwargs.pop("n_identities_per", 40), len(self.class_data.keys()))
        self.n_images_per = kwargs.pop("n_images_per", 25)
        self.n_eval_pairs = kwargs.pop("n_eval_pairs", 10000)

        # use these to cache our validation pairs
        self.eval_fps = []

    def get_train_batch(self, **kwargs):
        n_identities_per = kwargs.pop("n_identities_per", self.n_identities_per)
        n_images_per = kwargs.pop("n_images_per", self.n_images_per)
        names = list(self.class_data.keys())
        out_fps = []
        out_ids = []
        for _ in range(3):
            random.shuffle(names)
        for name in names[:n_identities_per]:
            fps = self.class_data[name]
            n_images = min(len(fps), n_images_per)
            for _ in range(3):
                random.shuffle(fps)
            for fp in fps[:n_images]:
                out_fps.append(fp)
                out_ids.append(self.name_to_idx[name])
        return np.asarray(out_fps), np.asarray(out_ids)

    def get_all_files(self):
        out_fps = []
        out_ids = []
        for idx, (class_name, file_paths) in enumerate(self.class_data.items()):
            for fp in file_paths:
                out_fps.append(fp)
                out_ids.append(self.name_to_idx[class_name])
        return out_fps, out_ids

    def generate_val(self):
        for identity, image_fps in self.class_data.items():
            all_negatives = []
            for iden2, neg_fps in self.class_data.items():
                if iden2 == identity:
                    continue
                all_negatives.extend(neg_fps)
            for fp1, fp2 in combinations(image_fps, 2):
                self.eval_fps.append([fp1, fp2, 1])
            for fp in image_fps:
                neg = random.choice(all_negatives)
                self.eval_fps.append([fp, neg, 0])

    def get_evaluation_batch(self, **kwargs):
        n_images_per = kwargs.pop("n_eval_pairs", self.n_eval_pairs)
        if len(self.eval_fps) == 0:
            self.generate_val()
        for _ in range(3):
            random.shuffle(self.eval_fps)
        out_fps = []
        for _ in range(n_images_per):
            idx = random.randint(0, len(self.eval_fps) - 1)
            out_fps.append(self.eval_fps.pop(idx))
            if len(self.eval_fps) == 0:
                break
        return np.asarray(out_fps)