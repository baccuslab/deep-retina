"""
Custom model classes

"""

from __future__ import absolute_import, division, print_function
from builtins import super
from os.path import join
from time import strftime

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

from preprocessing import datagen, loadexpt
from utils import notify, mksavedir, tocsv, tomarkdown, metric
from numpy.random import choice
from functools import partial

__all__ = ['Model', 'ln', 'convnet', 'lstm']


class Model(object):

    def __init__(self, cell_index, stimulus_type, loss, optimizer, mean_adapt):
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

        # save architecture as a json file
        self.savedir = mksavedir(prefix=str(self))
        with notify('Saving architecture'):
            with open(join(self.savedir, 'architecture.json'), 'w') as f:
                f.write(self.model.to_json())

        # function to write data to a CSV file
        self.save_csv = partial(tocsv, join(self.savedir, 'performance'))
        self.save_csv(['Epoch', 'Iteration', 'Training CC', 'Test CC'])

        # load experimental data
        self.stimulus_type = stimulus_type
        self.holdout = loadexpt(cell_index, self.stimulus_type, 'test', self.stim_shape[0], mean_adapt=mean_adapt)
        self.training = loadexpt(cell_index, self.stimulus_type, 'train', self.stim_shape[0], mean_adapt=mean_adapt)

        # save model information to a markdown file
        if 'architecture' not in self.__dict__:
            self.architecture = 'No architecture information specified'

        metadata = ['# ' + str(self), '## ' + strftime('%B %d, %Y'),
                    'Started training on: ' + strftime('%I:%M:%S %p'),
                    '### Architecture', self.architecture,
                    '### Stimulus', 'Experiment 10-07-15', stimulus_type, 'Mean adaptation: ' + str(mean_adapt),
                    'Cell #{}'.format(cell_index),
                    '### Optimization', str(loss), str(optimizer)]
        tomarkdown(join(self.savedir, 'README'), metadata)


    def train(self, batchsize, num_epochs=20, save_every=5):
        """
        Train the network!

        Parameters
        ----------
        batchsize : int

        num_epochs : int, optional
            Default: 20

        save_every : int, optional
            Default: 5

        """

        # initialize training iteration
        iteration = 0

        # loop over epochs
        for epoch in range(num_epochs):

            # save updates for this epoch
            res = self.test(epoch, iteration)

            # update display
            print('')
            print('='*20)
            print('==== Epoch #{:3d} ===='.format(epoch))
            print('='*20)
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
                print('{:05d}: {}'.format(iteration, loss))

    def predict(self, X):
        return self.model.predict(X)

    def test(self, epoch, iteration):

        # performance on the entire holdout set
        yhat_test = self.predict(self.holdout.X)
        corr_test = metric(yhat_test.ravel(), self.holdout.y)

        # performance on a subset of the training data
        training_sample_size = yhat_test.size
        inds = choice(self.training.y.size, training_sample_size, replace=False)
        yhat_train = self.predict(self.training.X[inds, ...])
        corr_train = metric(yhat_train.ravel(), self.training.y[inds])

        # save the results to a CSV file
        results = [epoch, iteration, corr_train, corr_test]
        self.save_csv(results)

        return results

    def save(self, epoch, iteration):
        """
        Save weights and optional test performance to directory

        """

        # store the weights
        filename = join(self.savedir, "epoch{:03d}_iter{:05d}_weights.h5".format(epoch, iteration))
        self.model.save_weights(filename)


class ln(Model):

    def __str__(self):
        return "LN"


    def __init__(self, cell_index, stimulus_type, loss='poisson_loss', optimizer='sgd',
                 weight_init='glorot_normal', l2_reg=0., mean_adapt=False):
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
            A Keras optimizer. Default: 'adam'

        weight_init : string or object, optional
            A Keras weight initialization method. Default: 'glorot_uniform'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """

        self.stim_shape = (40, 50, 50)

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
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt)


class convnet(Model):

    def __str__(self):
        return "convnet"

    def __init__(self, cell_index, stimulus_type, num_filters=(4, 16), filter_size=(9,9),
                 loss='poisson_loss', optimizer='adam', weight_init='normal', l2_reg=0., mean_adapt=False):
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

        self.stim_shape = (40, 50, 50)

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(Convolution2D(num_filters[0], filter_size[0], filter_size[1],
                                         input_shape=self.stim_shape, init=weight_init,
                                         border_mode='same', subsample=(1,1),
                                         W_regularizer=l2(l2_reg), activation='relu'))

            # max pooling layer
            self.model.add(MaxPooling2D(pool_size=(2, 2), ignore_border=True))

            # flatten
            self.model.add(Flatten())

            # Add dense (affine) layer with relu activation
            self.model.add(Dense(num_filters[1], init=weight_init, W_regularizer=l2(l2_reg), activation='relu'))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(Dense(1, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} convolutional filters of size {}'.format(num_filters[0], filter_size),
                                       '{} filters in the second (fully connected) layer'.format(num_filters[1]),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape)])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt)


class twolayer_convnet(Model):

    def __str__(self):
        return "two layer convnet"

    def __init__(self, cell_index, stimulus_type, num_filters=16, filter_size=(13,13),
                 loss='poisson_loss', optimizer='adam', weight_init='normal', l2_reg=0., mean_adapt=False):
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

        self.stim_shape = (40, 50, 50)

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(Convolution2D(num_filters, filter_size[0], filter_size[1],
                                         input_shape=self.stim_shape, init=weight_init,
                                         border_mode='same', subsample=(1,1),
                                         W_regularizer=l2(l2_reg), activation='relu'))

            # flatten
            self.model.add(Flatten())

            # Add a final dense (affine) layer with softplus activation
            self.model.add(Dense(1, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['{} convolutional filters of size {}'.format(num_filters, filter_size),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape)])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt)


class multilayer_convnet(Model):

    def __str__(self):
        return "multilayered_convnet"

    def __init__(self, cell_index, stimulus_type, conv_layers=[(12, 9, 9), (12, 9, 9)], dense_layer=64,
                 loss='poisson_loss', optimizer='adam', weight_init='normal', l2_reg=0., dropout=0.5, mean_adapt=False):
        """
        Multi-layered Convolutional neural network

        Parameters
        ----------

        cell_index : int
            Which cell to use

        stimulus_type : string
            Either 'whitenoise' or 'naturalscene'

        loss : string or object, optional
            A Keras objective. Default: 'poisson_loss'

        optimizer : string or object, optional
            A Keras optimizer. Default: 'adam'

        weight_init : string
            weight initialization. Default: 'normal'

        l2_reg : float, optional
            How much l2 regularization to apply to all filter weights

        """

        self.stim_shape = (40, 50, 50)

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # convolutional layers
            for ix, layer in enumerate(conv_layers):

                # get parameters for this layer
                num_filters, row_size, col_size = layer

                # convolutional layer
                if ix == 0:
                    self.model.add(Convolution2D(num_filters, row_size, col_size,
                                                input_shape=self.stim_shape, init=weight_init,
                                                border_mode='same', subsample=(1,1),
                                                W_regularizer=l2(l2_reg), activation='relu'))

                else:
                    self.model.add(Convolution2D(num_filters, row_size, col_size,
                                                input_shape=self.stim_shape, init=weight_init,
                                                border_mode='same', subsample=(1,1),
                                                W_regularizer=l2(l2_reg), activation='relu'))

                # max pooling layer
                self.model.add(MaxPooling2D(pool_size=(2, 2), ignore_border=True))

                # dropout
                self.model.add(Dropout(dropout))

            # flatten
            self.model.add(Flatten())

            # Add dense (affine) layer with relu activation
            self.model.add(Dense(dense_layer, init=weight_init, W_regularizer=l2(l2_reg), activation='relu'))
            self.model.add(Dropout(dropout))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(Dense(1, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))

        # save architecture string (for markdown file)
        self.architecture = '\n'.join(['Convolutional layers {}'.format(conv_layers),
                                       '{} filters in the second (fully connected) layer'.format(dense_layer),
                                       'weight initialization: {}'.format(weight_init),
                                       'l2 regularization: {}'.format(l2_reg),
                                       'stimulus shape: {}'.format(self.stim_shape)])

        # compile
        super().__init__(cell_index, stimulus_type, loss, optimizer, mean_adapt)
