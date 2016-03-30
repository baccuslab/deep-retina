"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train, generalizedconvnet, fixedlstm
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense
from keras.regularizers import l2


@main_wrapper
def fit_convnet(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(13, 13), weight_init='normal', l2_reg=0.01,
                     dropout=0.0)

    # compile the keras model
    model = sequential(layers, 'adam', loss='poisson')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = KerasMonitor('convnet', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


@main_wrapper
def fit_generalizedconvnet(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 2

    # get the convnet layers
    layers = generalizedconvnet(stim_shape, ncells, 
            architecture=('conv', 'noise', 'relu', 'conv', 'noise', 'relu', 'flatten', 'affine'),
            num_filters=[8, -1, -1, 16], filter_sizes=[15, -1, -1, 7], weight_init='normal',
            l2_reg=0.01, dropout=0.25, sigma=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = None #KerasMonitor('convnet', model, data, readme, save_every=20)

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
    monitor = KerasMonitor('fixedlstm', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model

@main_wrapper
def fit_fixedrnn(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    input_shape = (1000,16)
    ncells = len(cells)
    batchsize = 5000
    num_hidden = 100
    l2_reg = 0.01

    # add SimpleRNN layers
    layers = list()
    layers.append(SimpleRNN(num_hidden,
                            input_shape=input_shape,
                            return_sequences=False,
                            go_backwards=False,
                            init='glorot_uniform',
                            inner_init='orthogonal',
                            activation='tanh',
                            W_regularizer=l2(l2_reg),
                            U_regularizer=l2(l2_reg),
                            dropout_W=0.25,
                            dropout_U=0.25))

    # add dense layer
    layers.append(Dense(len(cells),
                        init='he_normal',
                        W_regularizer=l2(l2_reg),
                        activation='softplus'))

    #layers = fixedlstm(input_shape, len(cells), num_hidden=1600, weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize)

    # create a monitor to track progress
    monitor = KerasMonitor('fixedrnn', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':
    # list(range(37)) for 'all-cells'
    # [6,10,12,13] for '15-11-21a'
    # [0,2,7,10,11,12,31] for '16-01-07'
    # [0,3,7,9,11] for '16-01-08'
    gc_15_10_07 = [0,1,2,3,4]
    gc_15_11_21a = [6,10,12,13]
    gc_15_11_21b = [0,1,3,4,5,8,9,13,14,16,17,18,19,20,21,22,23,24,25]
    gc_16_01_07 = [0,2,7,10,11,12,31]
    gc_16_01_08 = [0,3,7,9,11]
    #mdl = fit_convnet(list(range(37)), 'naturalscene', 'all-cells')
    #mdl = fit_convnet([0,2,7,10,11,12,31], ['whitenoise', 'naturalscene', 'naturalmovie'], ['whitenoise', 'naturalscene', 'naturalmovie', 'structured'], '16-01-07')
    #mdl = fit_fixedlstm(list(range(37)), ['whitenoise_affine'], ['whitenoise_affine'], 'all-cells')
    #mdl = fit_convnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_fixedrnn(gc_15_10_07, ['whitenoise_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['naturalscene'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_convnet(gc_15_10_07, ['whitenoise', 'naturalscene'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_16_01_08, ['whitenoise', 'naturalscene', 'naturalmovie', 'whitenoise', 'naturalmovie', 'naturalmovie'], ['whitenoise', 'naturalscene', 'naturalmovie', 'structured'], '16-01-08')
    mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07')
