from triplet.params import ModelParams
from triplet.model import model_train
from utils import helper


def main():
    embedding_size = 128
    train_steps = 5000000

    checkpoint_dir = "checkpoints/triplet/train-subset-568226_{}_{}".format(embedding_size, helper.get_current_timestamp())
    checkpoint_dir = "checkpoints/triplet/train-subset-568226_128_2018-12-14-2025"

    small_train = 'fixtures/train-subset-568226.json'

    big_train = "fixtures/train-2876510.json"

    params = ModelParams(input_faces=small_train,
                         lfw="fixtures/lfw.json",
                         learning_rate=0.045,
                         identities_per_batch=40,
                         train_steps=train_steps,
                         checkpoint_dir=checkpoint_dir,
                         embedding_size=embedding_size,
                         loss_func='face_net',
                         pretrained_base_model=None,
                         use_center_loss=True,
                         batch_size=32)
    model_train(params)


if __name__ == "__main__":
    main()
