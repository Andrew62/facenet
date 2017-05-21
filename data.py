
import random
import helper
import numpy as np
import tensorflow as tf
from functools import partial
from inception_preprocessing import preprocess_image


def read_one_image(fname, **kwargs):
    """Reads one image given a filepath

    Parameters
    -----------
    fname : str
        path to a JPEG, PNG, or GIF file
    img_shape : tuple
        (kwarg) shape of the eventual image. Default is (224, 224, 3)
    is_training : bool
        (kwarg) boolean to tell the loader function if the graph is in training
        mode or testing. Default is True

    Returns
    -------
    preprocessed image
    """
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    is_training = kwargs.pop("is_training", False)
    # read the image file
    content = tf.read_file(fname)

    # decode buffer as an image
    img_raw = tf.image.decode_image(content, channels=img_shape[-1])

    return preprocess_image(img_raw, img_shape[0], img_shape[1], is_training=is_training)


class Dataset(object):
    def __init__(self, class_fp, **kwargs):
        self.class_data = helper.load_json(class_fp)
        self.idx_to_name = dict((idx, name) for idx, name in enumerate(sorted(self.class_data.keys())))
        self.name_to_idx = dict((name, idx) for idx, name in self.idx_to_name.items())
        self.n_identities_per = min(kwargs.pop("n_identities_per", 40), len(self.class_data.keys()))
        self.n_images_per = kwargs.pop("n_images_per", 25)

    def get_batch(self, **kwargs):
        n_identities_per = kwargs.pop("n_identities_per", self.n_identities_per)
        n_images_per = kwargs.pop("n_images_per", self.n_images_per)
        names = list(self.class_data.keys())
        out_fps = []
        out_ids = []
        for _ in range(3):
            random.shuffle(names)
        for name in names[:n_identities_per]:
            fps = self.class_data[name]
            n_images = min(len(fps), n_images_per)
            for _ in range(3):
                random.shuffle(fps)
            for fp in fps[:n_images]:
                out_fps.append(fp)
                out_ids.append(self.name_to_idx[name])
        return np.asarray(out_fps), np.asarray(out_ids)

    def get_all_files(self):
        out_fps = []
        out_ids = []
        for idx, (class_name, file_paths) in enumerate(self.class_data.items()):
            for fp in file_paths:
                out_fps.append(fp)
                out_ids.append(self.name_to_idx[class_name])
        return out_fps, out_ids


if __name__ == "__main__":
    dataset = Dataset("fixtures/faces.json")
    file_paths = tf.placeholder(tf.string)
    target_class = tf.placeholder(tf.int64)
    read = partial(read_one_image, is_training=False, image_shape=(224, 224, 3))
    images = tf.map_fn(read, file_paths, dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fp_list, id_list = dataset.get_batch(n_images_per=10000000)

        print("do it")
        imgs = sess.run(images, feed_dict={file_paths: fp_list,
                                                    target_class: id_list})
        print(imgs.shape)


