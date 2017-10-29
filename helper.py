import os
import time
import json
import tensorflow as tf


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


def initialize_uninitialized(sess):
    uninitialized_report = set(sess.run(tf.report_uninitialized_variables()))
    global_vars = tf.global_variables()
    to_initiazlize = []
    for var in global_vars:
        if var.name.split(":")[0] in uninitialized_report:
            print(var.name)
            to_initiazlize.append(var)
    sess.run(tf.variables_initializer(to_initiazlize))
