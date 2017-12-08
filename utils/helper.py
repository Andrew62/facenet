import os
import time
import json
import numpy as np


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


read_buffer_vect = np.vectorize(read_buffer)