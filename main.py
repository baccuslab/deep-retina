"""
Main script for training deep retinal models

"""

from __future__ import absolute_import

from keras.optimizers import Adam
from keras.objectives import poisson_loss

from models import ln, convnet, multilayer_convnet, twolayer_convnet, lstm
from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

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

def fit_multilayer_convnet(cell, stimulus_type):
    """
    Demo code for fitting a convnet model

    """

    # initialize model
    mdl = multilayer_convnet(cell, stimulus_type, conv_layers=[(12, 13, 13), (4, 9, 9)],
                             dense_layer=64, weight_init='normal', l2_reg=0.01,
                             dropout=0.2, mean_adapt=False)

    # train
    batchsize = 1000            # number of samples per batch
    num_epochs = 40             # number of epochs to train for
    save_weights_every = 100    # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


def fit_twolayer_convnet(cell, stimulus_type):
    """
    Demo code for fitting a convnet model

    """

    # initialize model
    mdl = twolayer_convnet(cell, stimulus_type, num_filters=4, filter_size=(13, 13),
                  weight_init='normal', l2_reg=0.01, mean_adapt=False)

    # train
    batchsize = 5000            # number of samples per batch
    num_epochs = 50             # number of epochs to train for
    save_weights_every = 50     # save weights every n iterations

    mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)

    return mdl


def fit_lstm(cell, stimulus_type, num_timesteps):
	mdl = lstm(cell, stimulus_type, num_timesteps=num_timesteps, num_filters=(8,16), filter_size=(13,13), loss='poisson_loss', weight_init='normal', l2_reg=0.01)
	batchsize = 100
	num_epochs = 50
	save_weights_every = 50
	mdl.train(batchsize, num_epochs=num_epochs, save_every=save_weights_every)
	return mdl


if __name__ == '__main__':
    pass
    # mdl = fit_lstm(4, 'naturalscene', 152)
    # mdl = fit_ln(0, 'whitenoise')
    # mdl = fit_convnet(0, 'naturalscene')
