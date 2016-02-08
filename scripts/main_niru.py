"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train
from deepretina.experiments import Experiment


def fit_convnet(cells, stimulus, exptdate='15-10-07'):
    """Demo code for fitting a convnet model"""

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 500

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(2, 4),
                     filter_size=(3, 3), weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize, load_fraction=0.1)

    # training options
    training_options = {
        'save_every': 5,           # save weights every n iterations
        'num_epochs': 10,           # number of epochs to train for
        'name': 'convnet',          # a name for the model
        'reduce_lr_every': 15       # halve the loss every n epochs
    }

    # train
    train(model, data, **training_options)

    return model


if __name__ == '__main__':
    mdl = fit_convnet([0, 1, 2, 3, 4], 'naturalscene')
