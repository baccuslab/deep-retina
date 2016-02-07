"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train
from deepretina.experiments import Experiment


def fit_convnet(reg, p, cells, stimulus, exptdate='15-10-07'):
    """Demo code for fitting a convnet model"""

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(13, 13), weight_init='normal', l2_reg=reg, dropout=p)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize)

    # training options
    training_options = {
        'save_every': 10,           # save weights every n iterations
        'num_epochs': 100,           # number of epochs to train for
        'name': 'convnet',          # a name for the model
        'reduce_lr_every': 15       # halve the loss every n epochs
    }

    # train
    train(model, data, **training_options)

    return model


if __name__ == '__main__':
	reg_arr = [0, 0.0001, 0.001, 0.01, 0.1]
	p_arr = [0, 0.25, 0.5, 0.75]
	for reg in reg_arr:
		for p in p_arr:
			mdl = fit_convnet(reg, p, [0, 1, 2, 3, 4], 'whitenoise')
