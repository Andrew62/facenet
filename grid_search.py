from train_triplet import model_train
from utils import helper
from itertools import product


class ModelParams(object):
    def __init__(self, **kwargs):
        self.input_faces = kwargs.pop("input_faces", "fixtures/faces.json")
        self.checkpoint_dir = kwargs.pop("checkpoint_dir")
        self.batch_size = 16
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.identities_per_batch = kwargs.pop("identities_per_batch", 100)
        self.n_images_per_iden = kwargs.pop("n_images_per_iden", 25)
        self.n_validation = kwargs.pop("n_validation", 10000)
        self.pretrained_base_model = "checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt"
        self.train_steps = kwargs.pop("train_steps", 40000)
        self.lfw = "fixtures/lfw.json"


def main():
#     learning_rates = [0.01, 0.001, 0.1]
#     identities_per_batch = [100, 200]

#     for lr, ipb in product(learning_rates, identities_per_batch):
    checkpoint_dir = "checkpoints/multiple_lr/" + helper.get_current_timestamp()
    params = ModelParams(learning_rate=0.01,
                         identities_per_batch=100,
                         train_steps=90000 * 2,
                         checkpoint_dir=checkpoint_dir,
                         embedding_size=200)
    model_train(params)


if __name__ == "__main__":
    main()
