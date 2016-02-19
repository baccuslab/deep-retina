"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train
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


if __name__ == '__main__':
    reg_arr = [0, 0.0001, 0.001, 0.01, 0.1]
    p_arr = [0, 0.25, 0.5, 0.75]
    for reg in reg_arr:
        for p in p_arr:
            mdl = fit_convnet([0, 1, 2, 3, 4], 'whitenoise', '15-10-07', reg, p,  description='WN convnet, l2_reg={}, dropout={}'.format(reg, p))
