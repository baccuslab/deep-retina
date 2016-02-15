"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train, generalizedconvnet
from deepretina.experiments import Experiment
from deepretina.io import Monitor, main_wrapper


@main_wrapper
def fit_convnet(cells, stimulus, exptdate, readme=None):
    """Main script for fitting a convnet
    
    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(13, 13), weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = Monitor('convnet', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model

@main_wrapper
def fit_generalizedconvnet(cells, stimulus, exptdate, readme=None):
    """Main script for fitting a convnet
    
    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 50

    # get the convnet layers
    layers = generalizedconvnet(stim_shape, ncells, 
            architecture=('conv', 'relu', 'conv', 'relu', 'flatten', 'affine'),
            num_filters=(8, -1, 25), filter_sizes=(13, -1, 13), weight_init='normal',
            l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = Monitor('convnet', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':
    #mdl = fit_convnet(list(range(37)), 'whitenoise', 'all-cells')
    mdl = fit_generalizedconvnet(list(range(37)), 'whitenoise', 'all-cells')
