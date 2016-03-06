"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, fixedlstm, train
from deepretina.experiments import Experiment
from deepretina.io import Monitor, main_wrapper


@main_wrapper
def fit_convnet(cells, stimulus, exptdate, l2_reg, dropout_probability, readme=None):
    """Main script for fitting a convnet

    author: Aran Nayebi
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(13, 13), weight_init='normal',
                     l2_reg=l2_reg, dropout=dropout_probability)

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
def fit_fixedlstm(cells, train_stimuli, test_stimuli, exptdate, timesteps, l2_reg, readme=None):
    """Main script for fitting a fixedlstm

    author: Aran Nayebi
    """

    input_shape = (timesteps, 64)
    ncells = len(cells)
    batchsize = 1000

    # get the fixedlstm layers
    layers = fixedlstm(input_shape, ncells, num_hidden=400, l2_reg=l2_reg)

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
    mdl = fit_fixedlstm(list(range(37)), ['naturalscenes_affine'], ['whitenoise_affine', 'naturalscenes_affine'], 'all-cells', 800, 0.01)
