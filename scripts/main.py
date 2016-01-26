"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from .models import ln, convnet, lstm
from keras.optimizers import RMSprop


def fit_ln(cell, stimulus_type):
    """
    Demo code for fitting an LN model in keras

    """

    # initialize model
    mdl = ln(cell, stimulus_type, l2_reg=0.01, optimizer='adam')

    # train
    batchsize = 5000            # number of samples per batch
    num_epochs = 15             # number of epochs to train for
    save_weights_every = 50     # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


def fit_convnet(cell, stimulus_type):
    """
    Demo code for fitting a convnet model

    """

    # initialize model
    mdl = convnet(cell, stimulus_type, num_filters=(8, 16), filter_size=(13, 13),
                  weight_init='normal', l2_reg=0.01, mean_adapt=False)

    # train
    batchsize = 5000            # number of samples per batch
    num_epochs = 10             # number of epochs to train for
    save_weights_every = 50     # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


def fit_lstm(cell, stimulus_type, num_timesteps):

    # modified optimizer
    RMSmod = RMSprop(lr=0.001, rho=0.99, epsilon=1e-6)

    # build the model
    mdl = lstm(cell, stimulus_type, num_timesteps=num_timesteps,
               num_filters=(8, 16), filter_size=(13, 13), loss='poisson_loss',
               optimizer=RMSmod, weight_init='he_normal', l2_reg=0.01)

    # training preferences
    batchsize = 100
    num_epochs = 150
    save_weights_every = 50

    # train
    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


if __name__ == '__main__':
    pass
    # mdl = fit_lstm(4, 'naturalscene', 152)
    # mdl = fit_ln(0, 'whitenoise')
    # mdl = fit_convnet([0,1,2,3,4], 'naturalscene')
    # mdl = fit_convnet(0, 'naturalscene')
