"""
Script to export squeeze layer network to accept image buffers as input
"""


import argparse
from export.export_classifier import export_classifier


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to a frozen graph")
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model folder")
    parser.add_argument("-e", "--embedding", type=int, required=True, help="size of the output embeddings")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="path for output file")

    args = parser.parse_args()

    export_classifier(args.embedding, args.model, args.output_file)
    print("done")


if __name__ == "__main__":
    main()
