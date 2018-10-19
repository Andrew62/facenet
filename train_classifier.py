from utils import helper
from classifier.model import train
from classifier.classification_args import ClassificationArgs


def main():
    args = ClassificationArgs(epochs=90,
                              checkpoint_dir="checkpoints/softmax/" + helper.get_current_timestamp(),
                              save_every=3000,
                              embedding_size=256,
                              train_csv="fixtures/train-subset-568226.csv",
                              batch_size=64,
                              learning_rate=0.01,
                              image_shape=(160, 160, 3),
                              num_classes=1708)  # for train-subset-568226.csv
    train(args)


if __name__ == "__main__":
    main()
