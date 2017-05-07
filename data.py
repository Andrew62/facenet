
import tensorflow as tf
from functools import partial
from inception_preprocessing import preprocess_image


def batch_producer(filepaths, **kwargs):
    """Function for loading batches of images and
    and labels from a csv *without* a header. CSV files
    must be in the format of
        /path/to/anchor/img,/path/to/positive/img,class_id
        /path/to/anchor/img,/path/to/positive/img,class_id
        /path/to/anchor/img,/path/to/positive/img,class_id

    Parameters
    -----------
    filepaths : list
        list of paths to csv files. Even if just using one file, it must
        be a list. For example ['/path/to/file.csv']
    batch_size : (kwarg) int
        number of samples per batch. Default is 4
    img_shape : (kwarg) tuple
        shape of the image. Must be in the form of (H,W,C). Image
        will *not* be resized, the value is used for setting
        the shape for the batch queue. Default is (224, 224, 3)
    is_training : (kwarg) bool
        when set to true, the loader will apply image transformations.
        Default is True
    num_threads : (kwarg) int
        number of threads to use for the loader. Default is 4
    
    Returns
    -------
    anchor_batch, positive_batch, class_id_batch
    """
    batch_size = kwargs.pop("batch_size", 4)
    img_shape = kwargs.pop("image_shape", (224, 224, 3))
    num_threads = kwargs.pop("num_threads", 4)
    is_training = kwargs.pop("is_trianing", True)

    # loads a series of text files
    filename_queue = tf.train.string_input_producer(filepaths)

    # used to read each text file line by line
    reader = tf.TextLineReader()

    # actually parse the text file. returns idx, content
    _, record = reader.read(filename_queue)

    # split out the csv. Defaults to returning strings. Input for this network
    # will be two images of the same identity and we'll randomly sample at
    # within a minibatch for a "hard" negative
    fp1, fp2, class_id = tf.decode_csv(record, record_defaults=[[""], [""], [""]])
    read_images = partial(read_one_image, is_training=is_training, image_shape=img_shape)
    content = [read_images(fp1), read_images(fp2), class_id]
    # load batches of images multithreaded. Use tf.stack and tf.unstack to push
    # pairs through
    anchor_batch, positive_batch, class_id_batch = tf.train.shuffle_batch(content,
                                                                          batch_size=batch_size,
                                                                          capacity=batch_size * 4,
                                                                          num_threads=num_threads,
                                                                          min_after_dequeue=batch_size * 2)
    return anchor_batch, positive_batch, class_id_batch


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
