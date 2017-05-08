import time

def get_current_timestamp():
    return time.strftime("%Y-%m-%d-%H%M", time.localtime())