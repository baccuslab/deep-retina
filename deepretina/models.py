"""
Construct and train deep neural network models using Keras

"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from .utils import notify
from . import io

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
    layers.append(Dense(nout, activation='softplus', init=weight_init,
                        W_regularizer=l2(l2_reg)))
    return layers


def convnet(input_shape, nout, num_filters=(8, 16), filter_size=(13, 13),
            weight_init='normal', l2_reg=0.0):
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

    # Add dense (affine) layer
    layers.append(Dense(num_filters[1], init=weight_init, W_regularizer=l2(l2_reg)))

    # Add relu activation
    layers.append(Activation('relu'))

    # Add a final dense (affine) layer with softplus activation
    layers.append(Dense(nout, init=weight_init,
                        W_regularizer=l2(l2_reg),
                        activation='softplus'))

    return layers


def train(model, data, save_every, num_epochs, name='model', reduce_lr_every=-1, reduce_rate=0.5):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model
        A Keras Model object

    data : experiments.Experiment
        An Experiment object

    save_every : int
        Saves the model parameters after `save_every` training batches.
        If save_every is less than or equal to zero, nothing gets saved.

    num_epochs : int
        Number of epochs to train for

    name : string
        A name for this model
    """
    assert isinstance(model, Model), "model is not a Keras model"
    assert isinstance(save_every, int), "save_every must be an integer"

    # create monitor for storing / saving model results
    if save_every > 0:
        monitor = io.Monitor(name, model, data)

    # initialize training iteration
    iteration = 0

    # loop over epochs
    try:
        for epoch in range(num_epochs):
            print('Epoch #{} of {}'.format(epoch + 1, num_epochs))

            # update learning rate on reduce_lr_every, assuming it is positive
            if (reduce_lr_every > 0) and (epoch > 0) and (epoch % reduce_lr_every == 0):
                lr = model.optimizer.lr.get_value()
                model.optimizer.lr.set_value(lr*reduce_rate)
                print(' Reduced learning rate to {} from {}'.format(lr*reduce_rate, lr))

            # loop over data batches for this epoch
            for X, y in data.batches(shuffle=True):

                # update on save_every, assuming it is positive
                if (save_every > 0) and (iteration % save_every == 0):
                    monitor.save(epoch, iteration)

                # train on the batch
                loss = model.train_on_batch(X, y)[0]

                # update
                iteration += 1
                print('  (Batch {} of {}) Loss: {}'.format(iteration, data.num_batches, loss))

    except KeyboardInterrupt:
        print('\nCleaning up')

    print('\nTraining complete!')


def fixedlstm(input_shape, nout, num_hidden=1600, weight_init='normal', l2_reg=0.0):
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

    # TODO: Add relu activation separately for threshold visualization
    # layers.append(Activation('relu', input_shape=input_shape))

    # Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
    layers.append(LSTM(num_hidden, forget_bias_init='one', return_sequences=True))

    # Add a final dense (affine) layer with softplus activation
    layers.append(TimeDistributedDense(nout, init=weight_init,
                                       W_regularizer=l2(l2_reg),
                                       activation='softplus'))

    return layers


def generalizedconvnet(input_shape, nout,
                       architecture=('conv', 'relu', 'pool', 'flatten', 'affine', 'relu', 'affine'),
                       num_filters=(4, -1, -1, -1, 16),
                       filter_sizes=(9, -1, -1, -1, -1),
                       weight_init='normal',
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

    filter_size : tuple, optional
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

    # initial convolutional layer
    layers.append(Convolution2D(num_filters[0], filter_sizes[0], filter_sizes[0],
                                input_shape=input_shape, init=weight_init,
                                border_mode='same', subsample=(1, 1), W_regularizer=l2(l2_reg)))

    for layer_id, layer_type in enumerate(architecture[1:]):

        # convolutional layer
        if layer_type == 'conv':
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

        # Add dense (affine) layer
        if layer_type == 'affine':
            layers.append(Dense(num_filters[layer_id], init=weight_init, W_regularizer=l2(l2_reg)))

        # Add a final dense (affine) layer with softplus activation
        if layer_type == 'finalaffine':
            layers.append(Dense(nout, init=weight_init,
                                W_regularizer=l2(l2_reg),
                                activation='softplus'))

    return layers
