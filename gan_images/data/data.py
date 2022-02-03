import pathlib

import tensorflow as tf
import tensorflow.keras as K


def load_real_samples(
    path : str = "../gan_images/data/thumbnails128x128",
    IMAGE_SIZE : int = 128,
    SEED : int = 32,
    batch_size : int = 64
):
    path = pathlib.Path(path)
    AUTOTUNE = tf.data.AUTOTUNE

    return K.utils.image_dataset_from_directory(
        path,
        labels = None,
        validation_split = 0,
        seed = SEED,
        batch_size = batch_size,
        image_size = (IMAGE_SIZE, IMAGE_SIZE)
    # the cache method keeps the dataset in memory after the first epoch to improve I/O costs
    # prefetch enables fetching the data while training to optimize computation time
    ).cache().prefetch(buffer_size = AUTOTUNE)

def init_rescale_image(IMAGE_SIZE : int):
    """
    1. Rescales images to the given size
    2. Maps inages float values into [0,1]

    :param IMAGE_SIZE: [description]
    :type IMAGE_SIZE: int
    :return: [description]
    :rtype: [type]
    """
    return K.Sequential([
        K.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        K.layers.Rescaling(1./256)
    ])



def data_augment(SEED):
    return K.Sequential([
        # Data augmentation is performed by randomly performing a vertical flip 
        K.layers.RandomFlip(mode = 'vertical', seed = SEED)
    ])


def data_preprocessing(
    IMAGES_SIZE : int = 128,
    SEED : int = 32
):
    return K.Sequential([
        init_rescale_image(IMAGES_SIZE),
        data_augment(SEED)
    ])