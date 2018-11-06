from utils import helper
from classifier.model import train
from classifier.classification_args import ClassificationArgs


def main():
    train_csvs = [
        "fixtures/train-partitions/0.part",
        "fixtures/train-partitions/1.part",
        "fixtures/train-partitions/2.part",
        "fixtures/train-partitions/3.part",
    ]
    small_train_csvs = ["fixtures/train-subset-568226.csv"],

    checkpoint_dir = "checkpoints/softmax_vgg/clipped_grads_128_" + helper.get_current_timestamp()
    args = ClassificationArgs(epochs=90,
                              checkpoint_dir=checkpoint_dir,
                              train_csvs=train_csvs,
                              num_classes=1708)  # for train-subset-568226.csv
    train(args)


if __name__ == "__main__":
    main()
