from utils import helper
from classifier.model import train
from classifier.classification_args import ClassificationArgs


def main():
    big_train_csvs = [
        "fixtures/train-partitions/0.part",
        "fixtures/train-partitions/1.part",
        "fixtures/train-partitions/2.part",
        "fixtures/train-partitions/3.part",
    ]
    big_train_n_classes = 8634

    small_train_csvs = [
        "fixtures/train-subset-568226.csv"
    ]
    small_train_n_classes = 1708

    checkpoint_dir = "checkpoints/softmax_vgg/inception_resnet_128_" + helper.get_current_timestamp()
    checkpoint_dir = "checkpoints/softmax_vgg/inception_resnet_128_2018-12-16-2336"

    image_shape = (182, 182, 3)

    lfw_json = "fixtures/lfw.json"
    args = ClassificationArgs(epochs=90,
                              checkpoint_dir=checkpoint_dir,
                              train_csvs=big_train_csvs,
                              num_classes=big_train_n_classes,
                              center_loss_reg_beta=0.004,
                              decay_epochs=1,
                              lfw_json=lfw_json,
                              image_shape=image_shape,
                              save_every=2)
    train(args)


if __name__ == "__main__":
    main()
