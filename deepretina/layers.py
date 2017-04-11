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

__all__ = ['ln', 'nips_conv', 'rgc_rnn']


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


def ln(x, nout, nln='softplus', weight_init='he_normal', l2_reg=0.01):
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
        Keras weight initialization (default: 'glorot_normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.01)
    """
    return Dense(nout, activation=nln, init=weight_init, kernel_regularizer=l2(l2_reg))(x)


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
    W_reg = [(0., l2_weight), (0., l2_weight)]
    act_reg = [(0., 0.), (0., 0.)]

    # loop over convolutional layers
    for (n, size), w_args, act_args in zip(convlayers, W_reg, act_reg):
        args = (n, size, size)
        kwargs = {
            'border_mode': 'valid',
            'subsample': (1, 1),
            'init': 'normal',
            'kernel_regularizer': l1_l2(*w_args),
            'activity_regularizer': l1_l2(*act_args),
        }
        if len(layers) == 0:
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


def conv(x, nfilters=8, sz=13, strides=(2, 2), l2reg=0.1):
    """Conv-BN-GaussianNoise-Relu"""
    y = Conv2D(nfilters, sz, strides=strides,
               kernel_regularizer=l2(l2reg),
               data_format="channels_first")(x)
    y = BatchNormalization()(y)
    y = GaussianNoise(0.05)(y)
    return Activation('relu')(y)


def rgc_cnn(x):
    return Model(inputs=x, outputs=Flatten(conv(conv(x))))


def rgc_rnn(xt, nout, rgc_cnn, state_size):
    x = Input(shape=xt.shape[2:])
    rnn = LSTM(state_size, stateful=True, return_sequences=True)(TimeDistributed(rgc_cnn(x))(xt))
    return TimeDistributed(Dense(nout))(rnn)


def bn_cnn(x, nout):
    l1 = conv(x)
    l2 = conv(l1)

    y = Concatenate()([Flatten()(l2), Flatten()(l1)])
    y = Dense(nout)(y)
    y = BatchNormalization()(y)
    return Activation('softplus')(y)


class ParametricSoftplus(Layer):
    def __init__(self, alpha_init=0.2, beta_init=5., **kwargs):
        """Parametric softplus nonlinearity

        Has the form: f(x) = alpha * log(1 + exp(beta * x))

        Parameters
        ----------
        alpha_init : array_like
            Initial values for the alphas (default: 0.2)

        beta_init : float
            Initial values for the betas (default: 5.0)
        """
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.beta_init = beta_init

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=input_shape[1:], initializer=Constant(self.alpha_init))
        self.beta = self.add_weight(shape=input_shape[1:], initializer=Constant(self.beta_init))
        super().build(input_shape)

    def call(self, inputs):
        return K.softplus(self.beta * inputs) * self.alpha

    def get_config(self):
        config = {}
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))
