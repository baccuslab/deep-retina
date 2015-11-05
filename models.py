"""
Custom model classes

"""

from __future__ import absolute_import

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

from preprocessing import datagen
from utils import notify


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
        self.model.compile(loss=loss, optimizer=optimizer)

    def load_data(self, cell_index, batchsize, **kwargs):
        """
        loads a generator that yields training data

        """

        self.data = datagen(cell_index, batchsize, history=self.history, **kwargs)

    def train(self, maxiter=1000):

        for k in range(maxiter):

            # load batch of data
            X, y = next(self.data)

            # train on the batch
            loss = self.model.train_on_batch(X, y)
            print('{:03d}: {}'.format(k, loss))

            # TODO: run callbacks


class ln(Model):

    def __init__(self, filter_shape, loss='poisson_loss', optimizer='sgd'):
        """
        Linear-nonlinear model with a parametric softplus nonlinearity

        Parameters
        ----------
        filter_shape : tuple
            shape of the linear filter (time, space, space)

        """

        # history of the filter
        self.history = filter_shape[0]

        # regularization
        l2_reg = 0.0

        # build the model (flatten the input, followed by a dense layer and
        # softplus activation)
        with notify('Building LN model'):
            self.model = Sequential()
            self.model.add(Flatten(input_shape=filter_shape))
            self.model.add(Dense(1, activation='softplus',
                                 W_regularizer=l2(l2_reg)))

        # compile
        with notify('Compiling'):
            super().__init__(loss, optimizer)


class convnet(Model):

    def __init__(self, filter_shape, loss='poisson_loss', optimizer='sgd'):
        """
        Convolutional neural network

        Parameters
        ----------
        history : int
            Number of steps in the history of the linear filter (Default: 40)

        """

        # history of the filter
        self.history = filter_shape[0]

        # regularization
        l2_reg = 0.0

        # build the model
        with notify('Building convnet'):

            self.model = Sequential()

            # first convolutional layer
            self.model.add(Convolution2D(16, 9, 9, input_shape=filter_shape, init='normal',
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
        with notify('Compiling'):
            super().__init__(loss, optimizer)


class lstm(Model):

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
        with notify('Compiling'):
            super().__init__(loss, optimizer)
