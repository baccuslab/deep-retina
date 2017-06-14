"""
Construct Keras models
"""
from __future__ import absolute_import, division, print_function
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from deepretina import activations

__all__ = ['bn_cnn', 'linear_nonlinear', 'ln', 'nips_cnn']


def bn_layer(x, nchan, size, l2_reg, sigma=0.05):
    """An individual batchnorm layer"""
    n = int(x.shape[-1]) - size + 1
    x = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    x = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(x)))
    return Activation('relu')(GaussianNoise(sigma)(x))


def bn_cnn(x, n_out, l2_reg=1.0):
    """Batchnorm CNN model"""
    x = bn_layer(x, 8, 15, l2_reg)
    x = bn_layer(x, 8, 11, l2_reg)
    x = Dense(n_out, use_bias=False)(Flatten()(x))
    return Activation('softplus')(BatchNormalization(axis=-1)(x))


def linear_nonlinear(x, n_out, activation='softplus', l2_reg=0.01):
    """A linear-nonlinear model"""

    # a default activation
    if activation in ('softplus', 'sigmoid', 'relu'):
        nonlinearity = Activation(activation)

    # one of the custom deepretina activations
    elif activation in activations.__all__:
        nonlinearity = activations.__dict__[activation]()

    # a custom class
    else:
        nonlinearity = activation

    x = Flatten()(x)
    x = Dense(n_out, kernel_regularizer=l2(l2_reg))(x)
    return nonlinearity(x)


def nips_cnn(x, n_out):
    """NIPS 2016 CNN Model"""
    # injected noise strength
    sigma = 0.1

    # first layer
    x = Conv2D(16, 15, data_format="channels_first", kernel_regularizer=l2(1e-3))(x)
    x = Activation('relu')(GaussianNoise(sigma)(x))

    # second layer
    x = Conv2D(8, 9, data_format="channels_first", kernel_regularizer=l2(1e-3))(x)
    x = Activation('relu')(GaussianNoise(sigma)(x))

    x = Flatten()(x)
    x = Dense(n_out, init='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(x)
    return Activation('softplus')(x)


# aliases
ln = linear_nonlinear
