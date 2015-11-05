"""
Main script for training deep retinal models

"""

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

from __future__ import absolute_import

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

from preprocessing import load_data
from models import ln, convnet, lstm


def fit_ln(cell=0)
    """
    Demo code for fitting an LN model in keras

    """

    # Train an LN model
    batchsize = 5000

    # initialize model, load data
    mdl = ln((40,50,50), optimizer='adam')
    mdl.load_data(cell, batchsize, filename='whitenoise')

    # train
    mdl.train(maxiter=500)

    return mdl


def fit_convnet(cell=0):
    """
    Demo code for fitting a convnet model

    """
    pass


def fit_lstm(cell=0):
    """
    Demo code for fitting an LSTM network

    """
    pass


if __name__ == '__main__':

    # fit an LN model to cell #0
    mdl = fit_ln()
