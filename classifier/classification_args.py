
class ClassificationArgs(object):
    def __init__(self, **kwargs):
        self.checkpoint_dir = kwargs["checkpoint_dir"]

        # should be a list of csvs. no header each line id,fp
        self.train_csvs = kwargs["train_csvs"]

        # number of unique classes
        self.num_classes = kwargs['num_classes']
        self.batch_size = kwargs.pop("batch_size", 32)
        self.image_shape = kwargs.pop("image_shape", (160, 160, 3))
        self.drop_out = kwargs.pop("drop_out", 0.8)
        self.center_loss_alpha = kwargs.pop("center_loss_alpha", 0.5)

        # face embedding sizecp ..
        self.embedding_size = kwargs.pop("embedding_size", 128)

        # epochs
        self.save_every = kwargs.pop('save_every', 5)
        self.epochs = kwargs.pop("epochs", 90)
        self.learning_rate = kwargs.pop("learning_rate", 0.045)
        self.regularization_beta = kwargs.pop("reg_beta", 4e-5)
