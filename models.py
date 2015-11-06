"""
Custom model classes

"""

from __future__ import absolute_import, division, print_function
from builtins import super
from os.path import join

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

from preprocessing import datagen
from utils import notify, mksavedir

__all__ = ['Model', 'ln', 'convnet', 'lstm']


class Model(object):

    def __init__(self, loss, optimizer):
        """
        Superclass for managing keras models

        Parameters
        ----------

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
        with notify('Saving architecture as json'):
            with open(join(self.savedir, 'architecture.json'), 'w') as f:
                f.write(self.model.to_json())

        # initialize training iteration
        self.iteration = 0
        self.epoch = 1

        # save initial weights
        self.save()

    def load_data(self, cell_index, batchsize, **kwargs):
        """
        loads a generator that yields training data

        """

        # load training data generator
        self.data = datagen(cell_index, batchsize, history=self.history, **kwargs)

        # loads data from h5 file, get the number of batches per epoch
        self.num_batches_per_epoch = next(self.data)

    def train(self, maxiter=1000, save_every=2):
        """
        Train the network!

        Parameters
        ----------
        maxiter : int, optional
            Number of iterations to run for (default: 1000)

        """

        try:
            for _ in range(maxiter):

                # update iteration
                self.iteration += 1

                # update epoch
                if self.iteration % self.num_batches_per_epoch == 0:
                    self.epoch += 1

                # load batch of data
                X, y = next(self.data)

                # train on the batch
                loss = self.model.train_on_batch(X, y)

                # update display and save
                print('{:03d}: {}'.format(self.iteration, loss))
                if self.iteration % save_every == 0:
                    self.save()

        except KeyboardInterrupt:
            with notify('Cleaning up'):
                self.save()

    def save(self):
        """
        Save weights to directory

        """
        filename = join(self.savedir, "epoch{:02d}_iter{:04d}_weights.h5".format(self.epoch, self.iteration))
        self.model.save_weights(filename)


class ln(Model):

    def __str__(self):
        return "LN"

    def __init__(self, stim_shape, loss='poisson_loss', optimizer='sgd'):
        """
        Linear-nonlinear model with a parametric softplus nonlinearity

        Parameters
        ----------
        stim_shape : tuple
            shape of the linear filter (time, space, space)

        """

        # history of the filter
        self.history = stim_shape[0]

        # regularization
        l2_reg = 0.0

        # build the model (flatten the input, followed by a dense layer and
        # softplus activation)
        with notify('Building LN model'):
            self.model = Sequential()
            self.model.add(Flatten(input_shape=stim_shape))
            self.model.add(Dense(1, activation='softplus',
                                 W_regularizer=l2(l2_reg)))

        # compile
        super().__init__(loss, optimizer)


class convnet(Model):

    def __str__(self):
        return "convnet"

    def __init__(self, stim_shape, num_filters=4, filter_size=(9,9), loss='poisson_loss', optimizer='sgd'):
        """
        Convolutional neural network

        Parameters
        ----------
        history : int
            Number of steps in the history of the linear filter (Default: 40)

        """

        # history of the filter
        self.history = stim_shape[0]

        # regularization
        l2_reg = 0.0

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(Convolution2D(num_filters, filter_size[0], filter_size[1],
                                         input_shape=stim_shape, init='normal',
                                         border_mode='same', subsample=(1,1),
                                         W_regularizer=l2(l2_reg), activation='relu'))

            # max pooling layer
            self.model.add(MaxPooling2D(pool_size=(2, 2), ignore_border=True))

            # flatten
            self.model.add(Flatten())

            # Add dense (affine) layer with relu activation
            self.model.add(Dense(32, init='normal', W_regularizer=l2(l2_reg), activation='relu'))

            # Add a final dense (affine) layer with softplus activation
            self.model.add(Dense(1, init='normal', W_regularizer=l2(l2_reg), activation='softplus'))

        # compile
        super().__init__(loss, optimizer)


class lstm(Model):

    def __str__(self):
        return "lstm"

    def __init__(self, history=40, loss='poisson_loss', optimizer='sgd'):
        """
        Convolutional neural network

        Parameters
        ----------
        history : int
            Number of steps in the history of the linear filter (Default: 40)

        """

        # history of the filter
        self.history = history

        # regularization
        l2_reg = 0.0

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # First layer is a time distributed convolutional layer
            self.model.add(Convolution2D(16, self.history, 9, 9, init='he_uniform',
                                         border_mode='full', subsample=(1,1),
                                         W_regularizer=l2(l2_reg), activation='relu'))


            # Second layer is a time distributed max pooling layer
            self.model.add(MaxPooling2D(poolsize=(2,2), ignore_border=True))

            # next we have a dense (affine) layer
            self.model.add(TimeDistributedDense(32, init='he_uniform',
                                                W_regularizer=l2(l2_reg), activation='relu'))

            # flatten -- is this necessary?
            # self.model.add(TimeDistributedFlatten())

            # add LSTM
            self.model.add(LSTM(32, init='he_uniform', forget_bias_init='one', activation='tanh', return_sequences=True))

            # add final layer
            self.model.add(Dense(1, init='he_uniform',
                           W_regularizer=l2(l2_reg), activation='softplus'))

        # compile
        super().__init__(loss, optimizer)
