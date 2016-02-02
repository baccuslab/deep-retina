"""
Classes for wrapping Keras models

"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from tqdm import tqdm
from .utils import notify
from .metrics import cc, lli, rmse
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


def train(model, data, save_every, num_epochs):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model
        A Keras Model object

    data : experiments.Experiment
        An Experiment object

    save_every : int
        Saves the model parameters after `save_every` training batches

    num_epochs : int
        Number of epochs to train for
    """

    assert isinstance(model, Model)

    # initialize training iteration
    iteration = 0

    # loop over epochs
    for epoch in range(num_epochs):

        # loop over data batches for this epoch
        for X, y in tqdm(data.batches('train', shuffle=True),
                         'Epoch #{}'.format(epoch + 1),
                         data.num_batches('train')):

            # update on save_every
            if iteration % save_every == 0:
                io.save(epoch, iteration)
                io.test(epoch, iteration)

            # update iteration
            iteration += 1

            # train on the batch
            loss = model.train_on_batch(X, y)

            # update display and save
            print('{:05d}: {}'.format(iteration, loss))

    # def test(self, epoch, iteration):

        # # performance on the entire holdout set
        # rhat_test = self.predict(self.holdout.X)
        # r_test = self.holdout.y

        # # performance on a subset of the training data
        # training_sample_size = rhat_test.shape[0]
        # inds = choice(self.training.y.shape[0], training_sample_size, replace=False)
        # rhat_train = self.predict(self.training.X[inds, ...])
        # r_train = self.training.y[inds]

        # # evalue using the given metrics  (computes an average over the different cells)
        # # ASSUMES TRAINING ON MULTIPLE CELLS
        # functions = map(multicell, (cc, lli, rmse))
        # results = [epoch, iteration]

        # for f in functions:
            # results.append(f(r_train, rhat_train)[0])
            # results.append(f(r_test, rhat_test)[0])

        # # save the results to a CSV file
        # self.save_csv(results)

        # # TODO: plot the train / test firing rates and save in a figure

        # return results

    # def save(self, epoch, iteration):
        # """
        # Save weights and optional test performance to directory

        # """

        # filename = join(self.weightsdir,
                        # "epoch{:03d}_iter{:05d}_weights.h5"
                        # .format(epoch, iteration))

        # # store the weights
        # self.model.save_weights(filename)


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
                       filter_sizes=(9,),
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
    raise NotImplementedError("generalized convnet is currently broken!")

    layers = list()

    # initial convolutional layer
    # layers.append(Convolution2D(num_filters

    for layer_id, layer_type in enumerate(architecture):

        # convolutional layer
        if layer_type == 'conv':
            layers.append(Convolution2D(num_filters[layer_id], filter_sizes[layer_id],
                                        filter_sizes[layer_id], input_shape=input_shape,
                                        init=weight_init, border_mode='same',
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
