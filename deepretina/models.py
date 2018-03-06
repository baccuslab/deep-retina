"""
Construct Keras models
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
keras = tf.keras
K = keras.backend
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Lambda = tf.keras.layers.Lambda
Reshape = tf.keras.layers.Reshape
Conv2D = tf.keras.layers.Conv2D
Permute = tf.keras.layers.Permute
ResizeMethod = tf.image.ResizeMethod
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
BatchNormalization = tf.keras.layers.BatchNormalization
GaussianNoise = tf.keras.layers.GaussianNoise
l1 = tf.keras.regularizers.l1
l2 = tf.keras.regularizers.l2
from deepretina import activations

__all__ = ['bn_cnn', 'linear_nonlinear', 'ln', 'nips_cnn']


def bn_layer(x, nchan, size, l2_reg, sigma=0.05):
    """An individual batchnorm layer"""
    n = int(x.shape[-1]) - size + 1
    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))
    return Activation('relu')(GaussianNoise(sigma)(y))

def resize_layer(x,resize):
    # account for channels_first
    xx = Permute((2,3,1))(x)
    xx = Lambda( lambda y: tf.image.resize_images(y, resize,
        method=ResizeMethod.NEAREST_NEIGHBOR))(x)
    return Permute((3,1,2))(xx)

def bn_layer_t(x, nchan, size, resize, l2_reg, sigma=0.05):
    """An individual batchnorm transpose-rescale layer"""
    y = resize_layer(x,resize)
    y = Conv2D(nchan, size, data_format="channels_first",
        padding='same',
        kernel_regularizer=l2(l2_reg))(y)
    # y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))
    return Activation('relu')(GaussianNoise(sigma)(y))

def bn_conv(x, nchan, size, l2_reg):
    """An individual batchnorm layer."""
    n = int(x.shape[-1]) - size + 1
    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))
    return Activation('relu')(y)

def bn_cnn(inputs, n_out, l2_reg=0.01):
    """Batchnorm CNN model"""
    y = bn_layer(inputs, 8, 15, l2_reg)
    y = bn_layer(y, 8, 11, l2_reg)
    y = Dense(n_out, use_bias=False)(Flatten()(y))
    outputs = Activation('softplus')(BatchNormalization(axis=-1)(y))
    return Model(inputs, outputs, name='BN-CNN')

def g_cnn(inputs, l2_reg=0.01,name="G-CNN"):
    """Batchnorm CNN model with ganglion convolution."""
    y = bn_conv(inputs, 8, 15, l2_reg)
    y = bn_conv(y, 8, 11, l2_reg)
    # y = bn_layer(y, 8, 9, l2_reg)
    y = Conv2D(8, 15, data_format="channels_first", kernel_regularizer=l2(l2_reg))(y)
    y = Activation('relu')(y)
    outputs = y
    return Model(inputs, outputs, name=name)

def auto_encoder(inputs, l2_reg=0.01,name="Autoencoder"):
    """Autoencoder CNN model with ganglion convolution."""
    encoded = bn_conv(inputs, 8, 15, l2_reg)
    encoded = bn_conv(encoded, 8, 11, l2_reg)
    encoded = Conv2D(8, 15, data_format="channels_first", kernel_regularizer=l2(l2_reg))(encoded)
    encoded = Activation('relu')(encoded)

    decoded = bn_layer_t(encoded, 8, 15, [26,26], l2_reg)
    decoded = bn_layer_t(decoded, 8, 11, [36,36], l2_reg)
    decoded = bn_layer_t(decoded, 1, 15, [50,50], l2_reg)
    decoded = Lambda(lambda y: tf.squeeze(y, 1))(decoded)
    return Model(inputs, (encoded, decoded), name=name)


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
