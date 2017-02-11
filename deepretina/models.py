"""
Construct Keras models
"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Graph
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.advanced_activations import PReLU, ParametricSoftplus
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.regularizers import l1l2, activity_l1l2, l2
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, SubunitSlice
from .utils import notify

__all__ = ['sequential', 'ln', 'convnet', 'fixedlstm', 'experimentallstm', 'nips_conv']


def sequential(layers, optimizer, loss='poisson'):
    """Compiles a Keras model with the given layers

    Parameters
    ----------
    layers : list
        A list of Keras layers, in order

    optimizer : string or optimizer
        Either the name of a Keras optimizer, or a Keras optimizer object

    loss : string, optional
        The name of a Keras loss function (Default: 'poisson_loss'), or a
        Keras objective object

    Returns
    -------
    model : keras.models.Sequential
        A compiled Keras model object
    """
    model = Sequential(layers)
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model

def ln(input_shape, nout, weight_init='glorot_normal', l2_reg=0.0):
    """A linear-nonlinear stack of layers

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (40,50,50))

    nout : int
        Number of output cells

    weight_init : string, optional
        Keras weight initialization (default: 'glorot_normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.0)
    """
    layers = list()
    layers.append(Flatten(input_shape=input_shape))
    layers.append(Dense(nout, init=weight_init, W_regularizer=l2(l2_reg)))
    layers.append(ParametricSoftplus())
    return layers


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
            'W_regularizer': l1l2(*w_args),
            'activity_regularizer': activity_l1l2(*act_args),
        }
        if len(layers) == 0:
            kwargs['input_shape'] = input_shape

        # add convolutional layer
        layers.append(Convolution2D(*args, **kwargs))

        # add gaussian noise
        layers.append(GaussianNoise(sigma))

        # add ReLu
        layers.append(Activation('relu'))

    # flatten
    layers.append(Flatten())

    # Add a final dense (affine) layer
    layers.append(Dense(num_cells, init='normal',
                        W_regularizer=l1l2(0., l2_weight),
                        activity_regularizer=activity_l1l2(1e-3, 0.)))

    # Finish it off with a parameterized softplus
    layers.append(ParametricSoftplus())

    return layers


def convnet(input_shape, nout,
            num_filters=(8, 16), filter_size=(13, 13),
            weight_init='normal',
            l2_reg_weights=(0.0, 0.0, 0.0),
            l1_reg_weights=(0.0, 0.0, 0.0),
            l2_reg_activity=(0.0, 0.0, 0.0),
            l1_reg_activity=(0.0, 0.0, 0.0),
            dropout=(0.0, 0.0)):
    """Convolutional neural network

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (40,50,50))

    nout : int
        Number of output cells

    num_filters : tuple, optional
        Number of filters in each layer. Default: (8, 16)

    filter_size : tuple, optional
        Convolutional filter size. Default: (13, 13)

    weight_init : string, optional
        Keras weight initialization (default: 'normal')

    l2_weights: tuple of floats, optional
        l2 regularization on the weights for each layer (default: 0.0)

    l2_activity: tuple of floats, optional
        l2 regularization on the activations for each layer (default: 0.0)

    dropout : tuple of floats, optional
        Fraction of units to 'dropout' for regularization (default: 0.0)
    """
    layers = list()

    def _regularize(layer_idx):
        """Small helper function to define per layer regularization"""
        return {
            'W_regularizer': l1l2(l1_reg_weights[layer_idx], l2_reg_weights[layer_idx]),
            'activity_regularizer': activity_l1l2(l1_reg_activity[layer_idx], l2_reg_activity[layer_idx]),
        }

    # first convolutional layer
    layers.append(Convolution2D(num_filters[0], filter_size[0], filter_size[1],
                                input_shape=input_shape, init=weight_init,
                                border_mode='valid', subsample=(1, 1),
                                **_regularize(0)))

    # Add relu activation
    layers.append(Activation('relu'))

    # max pooling layer
    layers.append(MaxPooling2D(pool_size=(2, 2)))

    # flatten
    layers.append(Flatten())

    # Dropout (first stage)
    layers.append(Dropout(dropout[0]))

    # Add dense (affine) layer
    layers.append(Dense(num_filters[1], init=weight_init, **_regularize(1)))

    # Add relu activation
    layers.append(Activation('relu'))

    # Dropout (second stage)
    layers.append(Dropout(dropout[1]))

    # Add a final dense (affine) layer
    layers.append(Dense(nout, init=weight_init, **_regularize(2)))

    # Finish it off with a parameterized softplus
    layers.append(ParametricSoftplus())

    return layers


def fixedlstm(input_shape, nout, num_hidden=1600, weight_init='he_normal', l2_reg=0.0):
    """LSTM network with fixed input (e.g. input from the CNN output)

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus e.g. (num_timesteps, stimulus.ndim)

    nout : int
        Number of output cells

    num_timesteps : int
        Number of timesteps of history to include in the LSTM layer

    num_filters : int, optional
        Number of filters in input. (Default: 16)

    num_hidden : int, optional
        Number of hidden units in the LSTM layer (Default: 1600)

    weight_init : string, optional
        Weight initialization for the final Dense layer (default: 'normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.0)
    """
    layers = list()

    # Optional: Add relu activation separately
    # layers.append(Activation('relu', input_shape=input_shape))

    # add the LSTM layer
    layers.append(LSTM(num_hidden, return_sequences=False, input_shape=input_shape))

    # Add a final dense (affine) layer with softplus activation
    layers.append(Dense(nout,
                        init=weight_init,
                        W_regularizer=l2(l2_reg),
                        activation='softplus'))

    return layers

def experimentallstm(input_shape, nout, num_hidden=50,
            num_filters=(8, 16), filter_size=(13, 13),
            optimizer='adam', loss='poisson_loss', lstm_flag=False, weight_init='normal', reg=0.01):
    """Experimental Convolutional-LSTM neural network

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (time, 40,50,50))

    nout : int
        Number of output cells

    num_filters : tuple, optional
        Number of filters in each layer. Default: (8, 16)

    filter_size : tuple, optional
        Convolutional filter size. Default: (13, 13)

    weight_init : string, optional
        Keras weight initialization (default: 'normal')

    reg: Just does l2 reg, can make this choice fancier and more modular later on
    """

    graph = Graph()

    # first define input
    graph.add_input(name='stim', input_shape=input_shape)

    #first convolutional layer
    graph.add_node(TimeDistributedConvolution2D(num_filters[0], filter_size[0], filter_size[1], init=weight_init, border_mode='valid', subsample=(1, 1), W_regularizer=l2(reg)), name='conv0', input='stim')

    # Add relu activation
    graph.add_node(Activation('relu'), name='r0', input='conv0')

    # max pooling layer
    graph.add_node(TimeDistributedMaxPooling2D(pool_size=(2, 2)), name='mp', input='r0')

    #Add an rnn per subunit
    rnn_arr = []
    for i in range(num_filters[0]):
        graph.add_node(SubunitSlice(i), name='slice'+str(i), input='mp')
        graph.add_node(TimeDistributedFlatten(), name='flatten'+str(i), input='slice'+str(i))
        if lstm_flag:
           graph.add_node(LSTM(num_hidden, return_sequences=False), name='rnn'+str(i), input='flatten'+str(i))
        else:
           graph.add_node(SimpleRNN(num_hidden, return_sequences=False), name='rnn'+str(i), input='flatten'+str(i))
        rnn_arr.append('rnn'+str(i))

    # Add first dense layer after concatenating RNN outputs
    graph.add_node(Dense(num_filters[1], init=weight_init, W_regularizer=l2(reg)), name='dense0', inputs=rnn_arr, merge_mode='concat')

    # Add relu activation
    graph.add_node(Activation('relu'), name='r1', input='dense0')
    
    # Add a final dense (affine) layer
    graph.add_node(Dense(nout, init=weight_init, W_regularizer=l2(reg)), name='dense1', input='r1')
    
    # Finish it off with a parametrized softplus
    graph.add_node(ParametricSoftplus(), name='softplus', input='dense1')
    
    graph.add_output(name='loss', input='softplus')

    with notify('Compiling'):
        graph.compile(optimizer, {'loss': loss})
    return graph
