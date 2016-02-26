"""
Construct and train deep neural network models using Keras

"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Model
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import ParametricSoftplus
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from .utils import notify

__all__ = ['sequential', 'train', 'ln', 'convnet', 'fixedlstm', 'generalizedconvnet']


def sequential(layers, optimizer, loss='poisson_loss'):
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
    model = Sequential()
    [model.add(layer) for layer in layers]
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


def convnet(input_shape, nout, num_filters=(8, 16), filter_size=(13, 13),
            weight_init='normal', l2_reg=0.0, dropout=0.0):
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

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.0)

    dropout : float, optional
        Fraction of units to 'dropout' for regularization (default: 0.0)

    """
    layers = list()

    # first convolutional layer
    layers.append(Convolution2D(num_filters[0], filter_size[0], filter_size[1],
                                input_shape=input_shape, init=weight_init,
                                border_mode='same', subsample=(1, 1),
                                W_regularizer=l2(l2_reg)))

    # Add relu activation
    layers.append(Activation('relu'))

    # max pooling layer
    layers.append(MaxPooling2D(pool_size=(2, 2)))

    # flatten
    layers.append(Flatten())

    # Dropout
    layers.append(Dropout(dropout))

    # Add dense (affine) layer
    layers.append(Dense(num_filters[1], init=weight_init, W_regularizer=l2(l2_reg)))

    # Add relu activation
    layers.append(Activation('relu'))

    # Add a final dense (affine) layer
    layers.append(Dense(nout, init=weight_init, W_regularizer=l2(l2_reg)))

    # Finish it off with a parameterized softplus
    layers.append(ParametricSoftplus())

    return layers


def train(model, data, monitor, num_epochs, reduce_lr_every=-1, reduce_rate=1.0):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model
        A Keras Model object

    data : experiments.Experiment
        An Experiment object

    monitor : io.Monitor
        Saves the model parameters and plots of performance progress

    num_epochs : int
        Number of epochs to train for

    reduce_lr_every : int
        How often to reduce the learning rate

    reduce_rate : float
        A fraction (constant) to multiply the learning rate by

    """
    assert isinstance(model, Model), "'model' must be a Keras model"

    # initialize training iteration
    iteration = 0

    # loop over epochs
    try:
        for epoch in range(num_epochs):
            print('Epoch #{} of {}'.format(epoch + 1, num_epochs))

            # update learning rate on reduce_lr_every, assuming it is positive
            if (reduce_lr_every > 0) and (epoch > 0) and (epoch % reduce_lr_every == 0):
                lr = model.optimizer.lr.get_value()
                model.optimizer.lr.set_value(lr * reduce_rate)
                print('\t(Changed learning rate to {} from {})'.format(lr * reduce_rate, lr))

            # loop over data batches for this epoch
            for X, y in data.train(shuffle=True):

                # update on save_every, assuming it is positive
                if (monitor is not None) and (iteration % monitor.save_every == 0):

                    # performs validation, updates performance plots, saves results to dropbox
                    monitor.save(epoch, iteration, y, model.predict(X))

                # train on the batch
                loss = model.train_on_batch(X, y)[0]

                # update
                iteration += 1
                print('{}\tLoss: {}'.format(iteration, loss))

    except KeyboardInterrupt:
        print('\nCleaning up')

    print('\nTraining complete!')


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


def generalizedconvnet(input_shape, nout,
                       architecture=('conv', 'relu', 'pool', 'flatten', 'affine', 'relu', 'affine'),
                       num_filters=(4, -1, -1, -1, 16),
                       filter_sizes=(9, -1, -1, -1, -1),
                       weight_init='normal',
                       dropout=0.0,
                       l2_reg=0.0):
    """Generic convolutional neural network

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

    num_filters : tuple, optional
        Number of filters in each layer. Default: [4, 16]

    filter_sizes : tuple, optional
        Convolutional filter size. Default: [9]
        Assumes that the filter is square.

    loss : string or object, optional
        A Keras objective. Default: 'poisson_loss'

    optimizer : string or object, optional
        A Keras optimizer. Default: 'adam'

    weight_init : string
        weight initialization. Default: 'normal'

    l2_reg : float, optional
        How much l2 regularization to apply to all filter weights

    """
    layers = list()

    for layer_id, layer_type in enumerate(architecture):

        # convolutional layer
        if layer_type == 'conv':
            if layer_id == 0:
                # initial convolutional layer
                layers.append(Convolution2D(num_filters[0], filter_sizes[0], filter_sizes[0],
                                            input_shape=input_shape, init=weight_init,
                                            border_mode='same', subsample=(1, 1), W_regularizer=l2(l2_reg)))
            else:
                layers.append(Convolution2D(num_filters[layer_id], filter_sizes[layer_id],
                                            filter_sizes[layer_id], init=weight_init, border_mode='same',
                                            subsample=(1, 1), W_regularizer=l2(l2_reg)))

        # Add relu activation
        if layer_type == 'relu':
            layers.append(Activation('relu'))

        # max pooling layer
        if layer_type =='pool':
            layers.append(MaxPooling2D(pool_size=(2, 2)))

        # flatten
        if layer_type == 'flatten':
            layers.append(Flatten())

        # dropout
        if layer_type == 'dropout':
            layers.append(Dropout(dropout))

        # batch normalization
        if layer_type == 'batchnorm':
            layers.append(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None))

        # Add dense (affine) layer
        if layer_type == 'affine':
            if layer_id == len(architecture) - 1:
                # add final affine layer with softplus activation
                layers.append(Dense(nout, init=weight_init,
                                    W_regularizer=l2(l2_reg),
                                    activation='softplus'))
            else:
                layers.append(Dense(num_filters[layer_id], init=weight_init, W_regularizer=l2(l2_reg)))

    return layers
