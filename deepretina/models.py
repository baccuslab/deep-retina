"""
Construct Keras models.
"""
from __future__ import absolute_import, division, print_function
from keras.models import Model
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
    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))
    return Activation('relu')(GaussianNoise(sigma)(y))


def bn_cnn(inputs, n_out, l2_reg=0.01):
    """Batchnorm CNN model"""
    y = bn_layer(inputs, 8, 15, l2_reg)
    y = bn_layer(y, 8, 11, l2_reg)
    y = Dense(n_out, use_bias=False)(Flatten()(y))
    outputs = Activation('softplus')(BatchNormalization(axis=-1)(y))
    return Model(inputs, outputs, name='BN-CNN')


def linear_nonlinear(inputs, n_out, *args, activation='softplus', l2_reg=0.01):
    """A linear-nonlinear model"""

    # a default activation
    if activation in ('softplus', 'sigmoid', 'relu', 'exp'):
        nonlinearity = Activation(activation)

    # is a nonlinearity class
    elif activation.lower() == ('rbf', 'psp'):
        nonlinearity = activations.__dict__[activation](*args)

    # one of the custom deepretina activations
    elif activation in activations.__all__:
        nonlinearity = activations.__dict__[activation]

    # a custom class
    else:
        nonlinearity = activation

    y = Flatten()(inputs)
    y = Dense(n_out, kernel_regularizer=l2(l2_reg))(y)
    outputs = nonlinearity(y)

    return Model(inputs, outputs, name=f'LN-{str(activation)}')


def nips_cnn(inputs, n_out):
    """NIPS 2016 CNN Model"""
    # injected noise strength
    sigma = 0.1

    # first layer
    y = Conv2D(16, 15, data_format="channels_first", kernel_regularizer=l2(1e-3))(inputs)
    y = Activation('relu')(GaussianNoise(sigma)(y))

    # second layer
    y = Conv2D(8, 9, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    y = Activation('relu')(GaussianNoise(sigma)(y))

    y = Flatten()(y)
    y = Dense(n_out, init='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    return Model(inputs, outputs, name='NIPS_CNN')


# aliases
ln = linear_nonlinear
