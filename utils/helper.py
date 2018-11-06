import os
import time
import json
import numpy as np
from typing import List
import math


def get_current_timestamp():
    return time.strftime("%Y-%m-%d-%H%M", time.localtime())


def load_json(fp):
    with open(fp) as infile:
        return json.load(infile)


def to_json(obj, fp):
    with open(fp, 'w') as target:
        json.dump(obj, target, indent=2)


def get_latest_dir(folder):
    sub_folders = []
    for item in os.listdir(folder):
        qualified = os.path.join(folder, item)
        if os.path.isdir(qualified):
            sub_folders.append(qualified)
    return list(reversed(sorted(sub_folders, key=os.path.getmtime)))[0]


def read_buffer(fp):
    with open(fp, 'rb') as inf:
        return inf.read()


def get_steps_per_epoch(fps: List[str], batch_size: int, header: bool=False) -> int:
    """
    Counter number of records in a csv and compute the number of training
    steps for one epoch. Rounds up!
    :param fps: list of files to process
    :param batch_size: number of examples per minibatch step
    :param header: indicates if files have a header or not
    :return: number of minibatches per epoch
    """
    n_lines = 0
    for fp in fps:
        with open(fp) as inf:
            if header:
                inf.readline()
            line = inf.readline()
            n_lines += 1
            while line:
                n_lines += 1
                line = inf.readline()
    return math.ceil(n_lines / batch_size)


read_buffer_vect = np.vectorize(read_buffer)
