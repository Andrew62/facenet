import os
import json
import argparse

def main():

    parser = argparse.ArgumentParser(description="Program to collect image paths and output a json file")
    parser.add_argument("-i", "--in_dir", help="top directory to walk", required=True,
                        type=str)
    parser.add_argument("-o", "--out_file", help="output json file",
                        default="images.json", dtype=str)
    parser.add_argument("-e", "--exts", help="file extensions to look for",
                        default=[".jpeg", ".gif", ".jpg", ".png"], nargs="+")
    args = parser.parse_args()

    # store all images by class
    images = {}

    # collect all images files first so we can sort
    # the class names alphabetically then get their index
    for d, _, files in os.walk(args.in_dir):
        for f in files:
            if any(map(lambda x: f.endswith(x), args.exts)):
                class_name = os.path.basename(d)
                if class_name not in images.keys():
                    images[class_name] = []
                images[class_name].append(os.path.join(os.path.abspath(d), f))

    # write out csv of all combinations of positives. Later we'll
    # sample negatives from within a minibatch
    with open(args.out_file, 'w', newline='') as target:
        json.dump(images, target, indent=2)

    print("done")

if __name__ == "__main__":
    main()
