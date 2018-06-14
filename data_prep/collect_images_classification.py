import os
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(description="Program to collect image paths and output a json file")
    parser.add_argument("-i", "--in_dir", help="top directory to walk", required=True,
                        type=str)
    parser.add_argument("-o", "--out_file", help="location to write output files",
                        type=str, default="faces.csv")
    parser.add_argument("-oi", "--out_identities", help="location to write output files",
                        type=str, default="face_identities.csv")
    parser.add_argument("-e", "--exts", help="file extensions to look for",
                        default=[".jpeg", ".gif", ".jpg", ".png"], nargs="+")
    parser.add_argument("-m", "--min_examples", help="minimum number of images per identity",
                        default=2, type=int)
    parser.add_argument("-x", "--exclude", help="directories to exclude",
                        nargs="+")
    args = parser.parse_args()

    # store all images by class
    images = []
    name_to_id = set()

    # collect all images files first so we can sort
    # the class names alphabetically then get their index
    for d, _, files in os.walk(args.in_dir):
        if any(map(lambda x: x in d, args.exclude)):
            print("Skipping " + d)
            continue
        for f in files:
            if any(map(lambda x: f.endswith(x), args.exts)):
                class_name = os.path.basename(d)
                name_to_id.add(class_name)
                images.append([os.path.join(os.path.abspath(d), f), class_name])

    name_to_id = sorted(name_to_id)
    with open(args.out_file, 'w') as target:
        writer = csv.writer(target)
        for (fp, name) in images:
            idx = name_to_id.index(name)
            writer.writerow([idx, fp])
    with open(args.out_identities, 'w') as target:
        writer = csv.writer(target)
        writer.writerows(name_to_id)

    print("Collected {0:,} images.".format(len(images)))


if __name__ == "__main__":
    main()
