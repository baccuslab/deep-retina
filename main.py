"""
Main script for training deep retinal models

"""

from __future__ import absolute_import

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

from models import ln, convnet


def fit_ln(cell, stimulus_type):
    """
    Demo code for fitting an LN model in keras

    """

    # initialize model
    mdl = ln(cell, stimulus_type, l2_reg=0.0)

    # train
    batchsize = 5000            # number of samples per batch
    num_epochs = 50             # number of epochs to train for
    save_weights_every = 10     # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


def fit_convnet(cell, stimulus_type):
    """
    Demo code for fitting a convnet model

    """

    # initialize model
    mdl = convnet(cell, stimulus_type, num_filters=(4, 16), filter_size=(13, 13),
                  weight_init='normal', l2_reg=0.01)

    # train
    batchsize = 5000            # number of samples per batch
    num_epochs = 50             # number of epochs to train for
    save_weights_every = 50     # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl

if __name__ == '__main__':

    # mdl = fit_ln(0, 'whitenoise')
    mdl = fit_convnet(0, 'naturalscene')
