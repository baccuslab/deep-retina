"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train, generalizedconvnet, fixedlstm
from deepretina.experiments import Experiment
from deepretina.io import Monitor, main_wrapper


@main_wrapper
def fit_convnet(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 64),
                     filter_size=(13, 13), weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

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
    batchsize = 1000

    # get the convnet layers
    layers = generalizedconvnet(stim_shape, ncells, 
            architecture=('conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'flatten', 'affine'),
            num_filters=[8, -1, 16, -1, 16, -1, 16, -1, 32], filter_sizes=[5, -1, 5, -1, 5, -1, 5, -1, 3], weight_init='normal',
            l2_reg=0.02)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = Monitor('convnet', model, data, readme, save_every=50)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


@main_wrapper
def fit_fixedlstm(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    input_shape = (1000,64)
    ncells = len(cells)
    batchsize = 1000

    # get the convnet layers
    layers = fixedlstm(input_shape, len(cells), num_hidden=1600, weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize)

    # create a monitor to track progress
    monitor = Monitor('fixedlstm', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':
    # list(range(37)) for 'all-cells'
    # [0,2,7,10,11,12,31] for '16-01-07'
    # [0,3,7,9,11] for '16-01-08'
    #mdl = fit_convnet(list(range(37)), 'naturalscene', 'all-cells')
    #mdl = fit_convnet([0,2,7,10,11,12,31], ['whitenoise', 'naturalscene', 'naturalmovie'], ['whitenoise', 'naturalscene', 'naturalmovie', 'structured'], '16-01-07')
    mdl = fit_fixedlstm(list(range(37)), ['whitenoise_affine'], ['whitenoise_affine'], 'all-cells')
