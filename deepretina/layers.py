"""
Custom Keras layers / stacks / models
"""
from __future__ import absolute_import, division, print_function

from keras import backend as K
from keras.engine import Layer
from keras.initializers import Constant
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Concatenate
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential
from keras.regularizers import l1_l2, l2

from .utils import notify

__all__ = ['linear_nonlinear', 'nips_conv', 'rgc_rnn']


def functional(inputs, outputs, optimizer, loss='poisson'):
    """Compiles a keras functional model

    Parameters
    ----------
    inputs: keras tensor
    outputs: keras tensor
    optimizer: Keras optimizer name or object
    loss: string, optional (default: poisson)
    """
    model = Model(inputs=inputs, outputs=outputs)
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model


def linear_nonlinear(x, nout, nln='softplus', weight_init='he_normal', l2_reg=0.01):
    """A linear-nonlinear model

    Parameters
    ----------
    x : Input
        Keras input tensor

    nout : int
        Number of output cells

    nln : string, optional
        default: 'softplus'

    weight_init : string, optional
        Keras weight initialization (default: 'he_normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.01)
    """
    assert nln in ('softplus', 'exp', 'sigmoid', 'relu')

    if nln == 'exp':
        nln = K.exp

    u = Dense(nout, init=weight_init, kernel_regularizer=l2(l2_reg))(x)
    return Activation(nln)(u)


def nips_conv(num_cells):
    """Hard-coded model for NIPS"""
    layers = list()
    input_shape = (40, 50, 50)

    # injected noise strength
    sigma = 0.1

    # convolutional layer sizes
    convlayers = [(16, 15), (8, 9)]

    # l2_weight_regularization for every layer
    l2_weight = 1e-3

    # weight and activity regularization
    weight_reg = ((0., l2_weight), (0., l2_weight))
    activity_reg = ((0., 0.), (0., 0.))

    # loop over convolutional layers
    for (n, size), w_args, act_args in zip(convlayers, weight_reg, activity_reg):
        args = (n, size, size)
        kwargs = {
            'border_mode': 'valid',
            'subsample': (1, 1),
            'init': 'normal',
            'kernel_regularizer': l1_l2(*w_args),
            'activity_regularizer': l1_l2(*act_args),
        }

        # if layers is empty, this is the first iteration, and we need to specify input shape
        if not layers:
            kwargs['input_shape'] = input_shape

        # add convolutional layer
        layers.append(Conv2D(*args, **kwargs))

        # add gaussian noise
        layers.append(GaussianNoise(sigma))

        # add ReLu
        layers.append(Activation('relu'))

    # flatten
    layers.append(Flatten())

    # Add a final dense (affine) layer
    layers.append(Dense(num_cells, init='normal',
                        kernel_regularizer=l1_l2(0., l2_weight),
                        activity_regularizer=l1_l2(1e-3, 0.)))

    # Finish it off with a parameterized softplus
    layers.append(Activation('softplus'))

    return layers


def conv(x, nfilters=8, kernel_size=13, strides=(2, 2), l2reg=0.1, sigma=0.05):
    """Conv-BN-GaussianNoise-Relu"""
    y = Conv2D(nfilters, kernel_size, strides=strides,
               kernel_regularizer=l2(l2reg),
               data_format="channels_first")(x)
    y = BatchNormalization()(y)
    y = GaussianNoise(sigma)(y)
    return Activation('relu')(y)


def rgc_cnn(x):
    return Model(inputs=x, outputs=Flatten(conv(conv(x))))


def rgc_rnn(xt, nout, u, state_size):
    rnn = LSTM(state_size, stateful=True, return_sequences=True)(TimeDistributed(u)(xt))
    return TimeDistributed(Dense(nout))(rnn)


def bn_cnn(x, nout):
    l1 = conv(x)
    l2 = conv(l1)

    y = Concatenate()([Flatten()(l2), Flatten()(l1)])
    y = Dense(nout)(y)
    y = BatchNormalization()(y)
    return Activation('softplus')(y)
