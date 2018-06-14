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
        self.pretrained_base_model = kwargs.pop("pretrained_base_model",
                                                "checkpoints/pretrained/inception_resnet_v2_2016_08_30.ckpt")
        self.train_steps = kwargs.pop("train_steps", 40000)
        self.lfw = "fixtures/lfw.json"
        self.loss_func = kwargs.pop("loss_func", "lossless")
        self.optimizer = kwargs.pop("optimizer", "adam")


def main():
    embedding_sizes = [256, 512]
    loss_funcs = ["face_net"]
    optimizers = ["adam"]
    train_steps = 50000 * 6

    for es, lf, opt in product(embedding_sizes, loss_funcs, optimizers):
        checkpoint_dir = "checkpoints/from_scratch/" + "{}_{}_{}_{}".format(lf, opt, es, train_steps) #helper.get_current_timestamp()
        params = ModelParams(learning_rate=0.01,
                             identities_per_batch=100,
                             train_steps=train_steps,
                             checkpoint_dir=checkpoint_dir,
                             embedding_size=es,
                             loss_func=lf,
                             optimizer=opt,
                             pretrained_base_model=None)
        model_train(params)


if __name__ == "__main__":
    main()
