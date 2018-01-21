from train import model_train
from utils import helper
from itertools import product


class ModelParams(object):
    def __init__(self, **kwargs):
        self.input_faces = kwargs.pop("input_faces", "fixtures/faces.json")
        self.checkpoint_dir = "checkpoints/inception_resnet_v2/" + helper.get_current_timestamp()
        self.batch_size = 80
        self.embedding_size = 128
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.identities_per_batch = kwargs.pop("identities_per_batch", 100)
        self.n_images_per_iden = kwargs.pop("n_images_per_iden", 25)
        self.n_validation = kwargs.pop("n_validation", 10000)
        self.pretrained_base_model = "checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt"
        self.train_steps = kwargs.pop("train_steps", 40000)
        self.decay_steps = kwargs.pop("decay_steps", 1000)
        self.decay_rate = kwargs.pop("decay_rate", 0.98)
        self.lfw = "fixtures/lfw.json"


def main():
    learning_rates = [0.0001, 0.001, 0.01]
    identities_per_batch = [100, 1000,  5000]
    train_steps = [2500, 35000, 40000]
    decay_steps = [250, 750, 1000, 1250]
    decay_rate = [0.92, 0.94, 0.96, 0.98]
    max_attempts = 3

    for lr, ipb, ts, ds, dr in product(learning_rates, identities_per_batch, train_steps, decay_steps, decay_rate):
        attempt = 0
        while attempt < max_attempts:
            try:
                params = ModelParams(learning_rate=lr,
                                     identities_per_batch=ipb,
                                     train_steps=ts,
                                     decay_steps=ds,
                                     decay_rate=dr)
                model_train(params)
                break   # made it!
            except Exception as e:
                print(e)
                attempt += 1


if __name__ == "__main__":
    main()
