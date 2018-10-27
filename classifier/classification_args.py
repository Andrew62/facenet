
class ClassificationArgs(object):
    def __init__(self, **kwargs):
        self.checkpoint_dir = kwargs["checkpoint_dir"]
        self.train_csv = kwargs["train_csv"]
        self.batch_size = kwargs.pop("batch_size", 32)
        self.image_shape = kwargs.pop("image_shape", (299, 299, 3))
        self.drop_out = kwargs.pop("drop_out", 0.8)
        self.center_loss_alpha = kwargs.pop("center_loss_alpha", 0.5)
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.save_every = kwargs.pop('save_every', 1000)
        self.epochs = kwargs.pop("epochs", 90)
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.num_classes = kwargs['num_classes']
        self.regularization_beta = kwargs.pop("reg_beta", 4e-5)
