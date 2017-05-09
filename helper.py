import os
import time


def get_current_timestamp():
    return time.strftime("%Y-%m-%d-%H%M", time.localtime())


def check_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)
