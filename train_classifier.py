from utils import helper
from classifier.model import train
from classifier.classification_args import ClassificationArgs


def main():
    args = ClassificationArgs(epochs=90,
                              checkpoint_dir="checkpoints/softmax_vgg/" "clipped_grads_" + helper.get_current_timestamp(),
                              save_every=10,  # epochs
                              embedding_size=256,
                              train_csv="fixtures/train-subset-568226.csv",
                              batch_size=32,
                              learning_rate=0.045,
                              image_shape=(160, 160, 3),
                              num_classes=1708,  # for train-subset-568226.csv
                              reg_beta=4e-5)
    train(args)


if __name__ == "__main__":
    main()
