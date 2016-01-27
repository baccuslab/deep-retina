"""
Custom model classes

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from builtins import super
from os.path import join
from time import strftime

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

from .preprocessing import datagen, loadexpt
from .utils import notify, Batch, mksavedir, tocsv, tomarkdown
from .metrics import cc, lli, rmse, multicell
from numpy.random import choice
from functools import partial

__all__ = ['Model', 'ln', 'convnet', 'lstm', 'fixedlstm']


class Model(object):

    def __init__(self, cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate):
        """
        Superclass for managing keras models

        Parameters
        ----------

        cell_index : int

        stimulus_type : string
            Either 'naturalscene' or 'whitenoise'

        loss : string or object, optional
            The loss function to use. (Default: poisson_loss)
            See http://keras.io/objectives/ for more information

        optimizer : string or object
            The optimizer to use. (Default: sgd)
            See http://keras.io/optimizers/ for more information

        """

        # compile the model
        with notify('Compiling'):
            self.model.compile(loss=loss, optimizer=optimizer)

        # create directories to save model results and weights in
        self.savedir = mksavedir(prefix=str(self))
        self.weightsdir = mksavedir(basedir='~/deep-retina-saved-weights/', prefix=str(self))

        # save architecture as a json file
        with notify('Saving architecture'):
            with open(join(self.savedir, 'architecture.json'), 'w') as f:
                f.write(self.model.to_json())

        # function to write data to a CSV file
        self.save_csv = partial(tocsv, join(self.savedir, 'performance'))
        self.save_csv(['Epoch', 'Iteration',
                       'CC:Train', 'CC:Test',
                       'LLI:Train', 'LLI:Test',
                       'RMSE:Train', 'RMSE:Test',
                       ])

        # load experimental data
        self.stimulus_type = stimulus_type
        load_data = partial(loadexpt, cell_index, self.stimulus_type,
                            history=self.stim_shape[0], exptdate=exptdate,
                            mean_adapt=mean_adapt)
        self.holdout = load_data('test')
        self.training = load_data('train')

        # save model information to a markdown file
        if 'architecture' not in self.__dict__:
            self.architecture = 'No architecture information specified'

        metadata = ['# ' + str(self), '## ' + strftime('%B %d, %Y'),
                    'Started training on: ' + strftime('%I:%M:%S %p'),
                    '### Architecture', self.architecture,
                    '### Stimulus', 'Experiment ' + exptdate, stimulus_type,
                    'Mean adaptation: ' + str(mean_adapt),
                    'Cell #{}'.format(cell_index),
                    '### Optimization', str(loss), str(optimizer)]

        # write metadata to a markdown file
        tomarkdown(join(self.savedir, 'README'), metadata)

    def train(self, batchsize, num_epochs=20, save_every=5):
        """
        Train the network!

        Parameters
        ----------
        batchsize : int

        num_epochs : int, optional
            Number of epochs to train for. (Default: 20)

        save_every : int, optional
            Saves the model parameters after `save_every` training batches. (Default: 5)

        """

        # initialize training iteration
        iteration = 0

        # loop over epochs
        for epoch in range(num_epochs):

            # save updates for this epoch
            res = self.test(epoch, iteration)

            # update display
            print('')
            print('=' * 20)
            print('==== Epoch #{:3d} ===='.format(epoch))
            print('=' * 20)
            print('Train CC: {:4.3f}'.format(res[2]))
            print(' Test CC: {:4.3f}\n'.format(res[3]))

            # loop over data batches for this epoch
            for X, y in datagen(batchsize, *self.training):

                # update on save_every
                if iteration % save_every == 0:
                    self.save(epoch, iteration)

                # update iteration
                iteration += 1

                # train on the batch
                loss = self.model.train_on_batch(X, y)

                # update display and save
                print('{:05d}: {}'.format(iteration, loss[0]))

    def predict(self, X):
        return self.model.predict(X)

    def test(self, epoch, iteration):

        # performance on the entire holdout set
        rhat_test = self.predict(self.holdout.X)
        r_test = self.holdout.y

        # performance on a subset of the training data
        training_sample_size = rhat_test.shape[0]
        inds = choice(self.training.y.shape[0], training_sample_size, replace=False)
        rhat_train = self.predict(self.training.X[inds, ...])
        r_train = self.training.y[inds]

        # evalue using the given metrics  (computes an average over the different cells)
        # ASSUMES TRAINING ON MULTIPLE CELLS
        functions = map(multicell, (cc, lli, rmse))
        results = [epoch, iteration]

        for f in functions:
            results.append(f(r_train, rhat_train)[0])
            results.append(f(r_test, rhat_test)[0])

        # save the results to a CSV file
        self.save_csv(results)

        # TODO: plot the train / test firing rates and save in a figure

        return results

    def save(self, epoch, iteration):
        """
        Save weights and optional test performance to directory

        """

        filename = join(self.weightsdir,
                        "epoch{:03d}_iter{:05d}_weights.h5"
                        .format(epoch, iteration))

        # store the weights
        self.model.save_weights(filename)


class ln(Model):

    def __str__(self):
        return "LN"

    def __init__(self, cell_index, stimulus_type, exptdate, loss='poisson_loss',
                 optimizer='sgd', weight_init='glorot_normal', l2_reg=0.,
                 mean_adapt=False, stimulus_shape=(40, 50, 50)):
        """
        Linear-nonlinear model with a parametric softplus nonlinearity

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

        loss : string or object, optional
            A Keras objective. Default: 'poisson_loss'

        optimizer : string or object, optional
            A Keras optimizer. Default: 'sgd'

        weight_init : string or object, optional
            A Keras weight initialization method. Default: 'glorot_uniform'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """

        self.stim_shape = stimulus_shape

        # build the model (flatten the input, followed by a dense layer and
        # softplus activation)
        with notify('Building LN model'):
            self.model = Sequential()
            self.model.add(Flatten(input_shape=self.stim_shape))
            self.model.add(Dense(1, activation='softplus', init=weight_init,
                                 W_regularizer=l2(l2_reg)))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape),
                                       'weight initialization: {}'.format(weight_init)])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate)


class convnet(Model):

    def __str__(self):
        return "convnet"

    def __init__(self, cell_index, stimulus_type, exptdate, num_filters=(4, 16),
                 filter_size=(9, 9), loss='poisson_loss', optimizer='adam',
                 weight_init='normal', l2_reg=0., mean_adapt=False,
                 stimulus_shape=(40, 50, 50)):
        """
        Convolutional neural network

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

        num_filters : tuple, optional
            Number of filters in each layer. Default: (4, 16)

        filter_size : tuple, optional
            Convolutional filter size. Default: (9, 9)

        loss : string or object, optional
            A Keras objective. Default: 'poisson_loss'

        optimizer : string or object, optional
            A Keras optimizer. Default: 'adam'

        weight_init : string
            weight initialization. Default: 'normal'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """

        self.stim_shape = stimulus_shape

        if type(cell_index) is int:
            # one output unit
            nout = 1

        else:
            # number of output units depends on the expt, hardcoded for now
            nout = len(cell_index)

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(Convolution2D(num_filters[0], filter_size[0], filter_size[1],
                                         input_shape=self.stim_shape, init=weight_init,
                                         border_mode='same', subsample=(1, 1),
                                         W_regularizer=l2(l2_reg)))

            # Add relu activation
            self.model.add(Activation('relu'))

            # max pooling layer
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # flatten
            self.model.add(Flatten())

            # Add dense (affine) layer
            self.model.add(Dense(num_filters[1], init=weight_init, W_regularizer=l2(l2_reg)))

            # Add relu activation
            self.model.add(Activation('relu'))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(Dense(nout, init=weight_init,
                                 W_regularizer=l2(l2_reg),
                                 activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} convolutional filters of size {}'.format(num_filters[0], filter_size),
                                       '{} filters in the second (fully connected) layer'.format(num_filters[1]),
                                       '{} output units'.format(nout),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape),
                                       ])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate)


class fixedlstm(Model):

    def __str__(self):
        return "fixedlstm"

    def __init__(self, cell_index, stimulus_type, exptdate, timesteps=152, num_filters=16, num_hidden=1600,
                 loss='poisson_loss', optimizer='adam', weight_init='normal', l2_reg=0., mean_adapt=False):
        """
        CNN-LSTM network with fixed CNN features as input.

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

        num_filters : optional
            Number of filters in input. Default: 16

        loss : string or object, optional
            A Keras objective. Default: 'poisson_loss'

        optimizer : string or object, optional
            A Keras optimizer. Default: 'adam'

        weight_init : string
            weight initialization. Default: 'normal'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """

        self.timesteps = timesteps
        if type(cell_index) is int:
            # one output unit
            nout = 1

        else:
            # number of output units depends on the expt, hardcoded for now
            nout = len(cell_index)

        # build the model
        with notify('Building fixedlstm'):

            self.model = Sequential()

            # Add relu activation separately for threshold visualization
            self.model.add(Activation('relu', input_shape=(timesteps, num_filters)))

            # Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
            self.model.add(LSTM(num_hidden, forget_bias_init='one', return_sequences=True))
#            self.model.add(LSTM(num_hidden, forget_bias_init='one', return_sequences=False))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(TimeDistributedDense(nout, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))
#            self.model.add(Dense(nout, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} output units'.format(nout),
                                       'weight initialization: {}'.format(weight_init), 'timesteps: {}'.format(self.timesteps),
                                       'hidden units: {}'.format(num_hidden), 'l2 regularization: {}'.format(l2_reg),
                                       'num input filters: {}'.format(num_filters),
                                       ])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate)


class lstm(Model):

    def __str__(self):
        return "lstm"

    def __init__(self, cell_index, stimulus_type, exptdate, num_timesteps=152, num_filters=(8, 16), filter_size=(13,13),
                 loss='poisson_loss', optimizer='adam', weight_init='normal', l2_reg=0., mean_adapt=False,
                 stimulus_shape=(40, 50, 50)):
        """
        LSTM

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

        num_filters : tuple, optional
            Number of filters in each layer. Default: (8, 16)

        num_timesteps: int
            Timesteps the recurrent layer should keep track of. Default: 152 ~1.9 seconds w/in 1-5 sec range

        filter_size : tuple, optional
            Convolutional filter size. Default: (13, 13)

        loss : string or object, optional
            A Keras objective. Default: 'poisson_loss'

        optimizer : string or object, optional
            A Keras optimizer. Default: 'adam'

        weight_init : string
            weight initialization. Default: 'normal'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """
        from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten

        self.stim_shape = (num_timesteps, stimulus_shape[0], stimulus_shape[1], stimulus_shape[2])

        # build the model
        with notify('Building lstm'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(TimeDistributedConvolution2D(num_filters[0], filter_size[0], filter_size[1],
                                                        input_shape=self.stim_shape, init=weight_init,
                                                        border_mode='same', subsample=(1, 1),
                                                        W_regularizer=l2(l2_reg)))

            # Add relu activation separately
            self.model.add(Activation('relu'))

            # max pooling layer
            self.model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2)))

            # flatten
            self.model.add(TimeDistributedFlatten())

            # Add dense (affine) layer with relu activation
            self.model.add(TimeDistributedDense(num_filters[1], init=weight_init, W_regularizer=l2(l2_reg)))

            # Add relu activation separately for threshold visualization
            self.model.add(Activation('relu'))

            # Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
            self.model.add(LSTM(100 * num_filters[1], forget_bias_init='one', return_sequences=True))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(TimeDistributedDense(1, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} convolutional filters of size {}'.format(num_filters[0], filter_size),
                                       '{} filters in the second (fully connected) layer'.format(num_filters[1]),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape)])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate)

        # hack to fix train/test datasets for use with the LSTM architecture
        numTime = self.stim_shape[0]
        self.holdout = loadexpt(cell_index, self.stimulus_type, 'test', self.stim_shape[1], mean_adapt=mean_adapt)
        self.training = loadexpt(cell_index, self.stimulus_type, 'train', self.stim_shape[1], mean_adapt=mean_adapt)
        X_train = self.training.X
        y_train = self.training.y
        X_test = self.holdout.X
        y_test = self.holdout.y
        numTrain = (int(X_train.shape[0]/numTime))*numTime
        numTest = (int(X_test.shape[0]/numTime))*numTime
        X_train = X_train[:numTrain]
        y_train = y_train[:numTrain]
        X_test = X_test[:numTest]
        y_test = y_test[:numTest]
        X_train = np.reshape(X_train, (int(numTrain/numTime), numTime, self.stim_shape[1], self.stim_shape[2], self.stim_shape[3]))
        y_train = np.reshape(y_train, (int(numTrain/numTime), numTime, 1))
        X_test = np.reshape(X_test, (int(numTest/numTime), numTime, self.stim_shape[1], self.stim_shape[2], self.stim_shape[3]))
        y_test = np.reshape(y_test, (int(numTest/numTime), numTime, 1))
        self.training = Batch(X_train, y_train)
        self.holdout = Batch(X_test, y_test)


class generalizedconvnet(Model):

    def __str__(self):
        return "generalizedconvnet"

    def __init__(self, cell_index, stimulus_type, exptdate,
                 layers=['conv', 'relu', 'pool', 'flatten', 'affine', 'relu', 'finalaffine'],
                 num_filters=[4, -1, -1, -1, 16], filter_sizes=[9], loss='poisson_loss', optimizer='adam',
                 weight_init='normal', l2_reg=0.01, mean_adapt=False, stimulus_shape=(30, 50, 50)):
        """
        Convolutional neural network

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

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

        self.stim_shape = stimulus_shape

        if type(cell_index) is int:
            # one output unit
            nout = 1

        else:
            # number of output units depends on the expt, hardcoded for now
            nout = len(cell_index)

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            for layer_id, layer_type in enumerate(layers):

                if layer_type == 'conv':
                    # convolutional layer
                    self.model.add(Convolution2D(num_filters[layer_id], filter_sizes[layer_id],
                                                 filter_sizes[layer_id], input_shape=self.stim_shape,
                                                 init=weight_init, border_mode='same',
                                                 subsample=(1, 1), W_regularizer=l2(l2_reg)))
                if layer_type == 'relu':
                    # Add relu activation
                    self.model.add(Activation('relu'))

                if layer_type =='pool':
                    # max pooling layer
                    self.model.add(MaxPooling2D(pool_size=(2, 2)))

                if layer_type == 'flatten':
                    # flatten
                    self.model.add(Flatten())

                if layer_type == 'affine':
                    # Add dense (affine) layer
                    self.model.add(Dense(num_filters[layer_id], init=weight_init, W_regularizer=l2(l2_reg)))

                if layer_type == 'finalaffine':
                    # Add a final dense (affine) layer with softplus activation
                    self.model.add(Dense(nout, init=weight_init,
                                         W_regularizer=l2(l2_reg),
                                         activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} layers'.format(layers),
                                       '{} number of filters'.format(num_filters),
                                       '{} size of filters'.format(filter_sizes),
                                       '{} output units'.format(nout),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape),
                                       ])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt, exptdate)
