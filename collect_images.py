import os
import csv
from itertools import combinations

# input directory where each subfolder is the unique name of the
# person. Will look for jpegs, gifs, and pngs
base_dir = "fixtures/faces"

# output csv for the training input producer
out_csv = "fixtures/faces.csv"

# store all images by class
images = {}

# these are the images exts that tf.decode_image can handle
exts = [".jpeg", ".gif", ".jpg", ".png"]

# collect all images files first so we can sort
# the class names alphabetically then get their index
for d, _, files in os.walk(base_dir):
    for f in files:
        if any(map(lambda x: f.endswith(x), exts)):
            class_name = os.path.basename(d)
            if class_name not in images.keys():
                images[class_name] = []
            images[class_name].append(os.path.join(os.path.abspath(d), f))

# write out csv of all combinations of positives. Later we'll
# sample negatives from within a minibatch
with open(out_csv, 'w', newline='') as target:
    writer = csv.writer(target)
    classes = sorted(images.keys())
    for name, fps in images.items():
        for anchor, positive in combinations(fps, 2):
            writer.writerow([anchor, positive, classes.index(name)])

print("done")
