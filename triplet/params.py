

class ModelParams(object):
    def __init__(self, **kwargs):
        self.checkpoint_dir = kwargs["checkpoint_dir"]
        self.input_faces = kwargs.pop("input_faces", "fixtures/faces.json")
        self.batch_size = kwargs.pop("batch_size", 32)
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.learning_rate = kwargs.pop("learning_rate", 0.045)
        self.identities_per_batch = kwargs.pop("identities_per_batch", 100)
        self.n_images_per_iden = kwargs.pop("n_images_per_iden", 25)
        self.n_validation = kwargs.pop("n_validation", 1000)
        self.train_steps = kwargs.pop("train_steps", 40000)
        self.lfw = "fixtures/lfw.json"
        self.loss_func = kwargs.pop("loss_func", "facenet_loss")
        self.center_loss_alpha = kwargs.pop("center_loss_alpha", 0.5)
        self.use_center_loss = kwargs.pop("use_center_loss", True)
        self.regularization_beta = kwargs.pop("reg_beta", 4e-5)
        self.drop_out = kwargs.pop("drop_out", 0.8)
        self.image_shape = kwargs.pop("image_shape", (160, 160, 3))

