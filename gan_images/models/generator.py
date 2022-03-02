import tensorflow as tf
from tensorflow import keras as K


def init_dense(input_shape, image_dim, D):
    """[summary]

    :param input_shape: [description]
    :type input_shape: [type]
    :param image_dim: [description]
    :type image_dim: [type]
    :param D: [description]
    :type D: [type]
    :return: [description]
    :rtype: [type]
    """
    return K.layers.Dense(
        units = image_dim[0] * image_dim[1] * D, # dim of the output space
        use_bias = False,
        input_shape = (input_shape, )
    )

def init_conv2D_transpose(filter_dim, D, strides, activation = None):
    """[summary]

    :param filter_dim: [description]
    :type filter_dim: [type]
    :param D: [description]
    :type D: [type]
    :param strides: [description]
    :type strides: [type]
    :param activation: [description], defaults to None
    :type activation: [type], optional
    :return: [description]
    :rtype: [type]
    """
    return K.layers.Conv2DTranspose(
            filters = D,
            kernel_size = (filter_dim, filter_dim),
            strides = strides,
            padding = 'same',
            use_bias = False,
            activation = activation
        )

def init_generator(
    image_shape = (28, 28), 
    D = 256,
    filter_dim = 5,
    num_channels = 3
):
    """[summary]

    :param image_shape: [description], defaults to (28, 28)
    :type image_shape: tuple, optional
    :param D: [description], defaults to 256
    :type D: int, optional
    :param filter_dim: [description], defaults to 5
    :type filter_dim: int, optional
    :param num_channels: number of image channels to generate (3 for RGB), defaults to 3
    :type num_channels: int, optional 
    :return: [description]
    :rtype: [type]
    """
    # Init the model
    model = K.Sequential()

    shape_one = (int(image_shape[0] / 4), int(image_shape[1] / 4))

    # The first layer transforms the seed into a flat tensor
    # It takes as input the seed and returns a flat tensor of 
    model.add(init_dense(input_shape = 100, image_dim = shape_one, D = D))

    # The BatchNormalization layer keeps the mean close to 0 and the std to 1 for the output.
    # While training (i.e. using fit()), it will normalize on every batch gamma * ((B - mean(B) / sqrt(variance(B) + eps) + beta)
    # Need to look more deep into Virtual Batch Normalization
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU(alpha = 0.3))
    model.add(K.layers.Reshape((shape_one[0], shape_one[1], D)))

    model.add(init_conv2D_transpose(filter_dim=filter_dim, D = int(D/2), strides = (1,1)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU(alpha = 0.3))

    model.add(init_conv2D_transpose(filter_dim=filter_dim, D = int(D/4), strides = (2,2)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.LeakyReLU(alpha = 0.3))

    model.add(init_conv2D_transpose(filter_dim=filter_dim, D = num_channels, strides = (2,2), activation = 'tanh'))

    return model

def g_loss(fake):
    # The from_logits = True indicates the values are in [-inf, inf]
    cross_entropy = K.losses.BinaryCrossentropy(from_logits = True)
    return cross_entropy(tf.ones_like(fake), fake)