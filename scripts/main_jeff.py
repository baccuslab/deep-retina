"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, generalizedconvnet, fixedlstm
from deepretina.core import train
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
def fit_generalizedconvnet(cells, train_stimuli, test_stimuli, exptdate, nclip=0, readme=None, sigma=0.1, num_filters=(8,16)):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 8000

    # get the convnet layers
    #### BEST CONV-CONV-AFFINE ARCHITECTURE ####
    layers = generalizedconvnet(stim_shape, ncells,
            architecture=('conv', 'noise', 'relu', 'conv', 'noise', 'relu', 'flatten', 'affine', 'param_softplus'),
            num_filters=[num_filters[0], -1, -1, num_filters[1]], filter_sizes=[15, -1, -1, 7], weight_init='normal',
            l2_reg=0.05, dropout=0.25, sigma=sigma)
    
    #### BEST CONV-AFFINE-AFFINE ARCHITECTURE ####
    #layers = generalizedconvnet(stim_shape, ncells, 
    #        architecture=('conv', 'requ', 'batchnorm', 'flatten', 'dropout', 'affine', 'requ', 'batchnorm', 'flatten', 'affine'),
    #        num_filters=[8, -1, -1, -1, -1, 16], filter_sizes=[15], weight_init='normal',
    #        l2_reg=0.01, dropout=0.5)

    #layers = generalizedconvnet(stim_shape, ncells, 
    #        architecture=('conv', 'noise', 'relu', 'flatten', 'dropout', 'affine', 'noise', 'relu', 'affine', 'param_softplus'),
    #        num_filters=[num_filters[0], -1, -1, -1, -1, num_filters[1]], filter_sizes=[17], weight_init='normal',
    #        l2_reg=0.02, dropout=0.25, activityl1=1e-3, sigma=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='poisson')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize, nskip=nclip, zscore_flag=True)

    # create a monitor to track progress
    monitor = KerasMonitor('convnet', model, data, readme, save_every=20)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


@main_wrapper
def fit_fixedlstm(cells, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    input_shape = (1000,32)
    ncells = len(cells)
    batchsize = 250

    # get the convnet layers
    layers = fixedlstm(input_shape, len(cells), num_hidden=200, weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam', loss='poisson')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize, nskip=0, zscore_flag=False)

    # create a monitor to track progress
    monitor = KerasMonitor('fixedlstm', model, data, readme, save_every=20)

    # train
    train(model, data, monitor, num_epochs=100)

    return model

@main_wrapper
def fit_fixedrnn(cells, train_stimuli, test_stimuli, exptdate, readme=None, num_affine=16):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    input_shape = (1000,num_affine)
    ncells = len(cells)
    batchsize = 4000
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
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize, zscore_flag=False, nskip=0)

    # create a monitor to track progress
    monitor = KerasMonitor('fixedrnn', model, data, readme, save_every=20)

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
    #gc_16_05_31 = [2,3,4,9,10,11,12,14,16,17,18,20,25,27]
    gc_16_05_31 = [2,3,4,14,16,18,20,25,27]
    #mdl = fit_convnet(list(range(37)), 'naturalscene', 'all-cells')
    #mdl = fit_convnet([0,2,7,10,11,12,31], ['whitenoise', 'naturalscene', 'naturalmovie'], ['whitenoise', 'naturalscene', 'naturalmovie', 'structured'], '16-01-07')
    #mdl = fit_fixedlstm(list(range(37)), ['whitenoise_affine'], ['whitenoise_affine'], 'all-cells')
    #mdl = fit_convnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_fixedrnn(gc_15_10_07, ['whitenoise_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['naturalscene'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_convnet(gc_15_10_07, ['whitenoise', 'naturalscene'], ['whitenoise', 'naturalscene'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_16_01_08, ['whitenoise', 'naturalscene', 'naturalmovie', 'whitenoise', 'naturalmovie', 'naturalmovie'], ['whitenoise', 'naturalscene', 'naturalmovie', 'structured'], '16-01-08')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise_3_31_2016'], ['whitenoise_3_31_2016', 'naturalscene_3_31_2016'], '15-10-07', nclip=5000)
    #mdl = fit_fixedrnn(gc_15_10_07, ['naturalscenes_affine_c82720'], ['whitenoise_affine_3dd884', 'naturalscenes_affine_c82720'], '15-10-07')
    #mdl = fit_fixedlstm(gc_15_10_07, ['naturalscenes_affine_c82720'], ['whitenoise_affine_3dd884', 'naturalscenes_affine_c82720'], '15-10-07')
    #mdl = fit_fixedrnn(gc_15_10_07, ['naturalscene_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-10-07')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=6000)
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise_augmented_3x'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=6000, description='conv-affine-affine version of 3520cd on whitenoise')
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['naturalscene_augmented_3x'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=6000, description='conv-affine-affine version of 3520cd on naturalscene')
    #mdl = fit_fixedlstm(gc_15_10_07, ['naturalscene_affine_007c52'], ['whitenoise_affine_9a1b0c', 'naturalscene_affine_007c52'], '15-10-07', description='fixedlstm naturalscene on 007c52')
    #mdl = fit_generalizedconvnet(gc_15_11_21a, ['naturalscene'], ['whitenoise', 'naturalscene'], '15-11-21a', nclip=6000, description='conv-affine-affine model on 15-11-21a naturalscene')
    #mdl = fit_generalizedconvnet(gc_15_11_21a, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-11-21a', nclip=6000, description='conv-affine-affine model on 15-11-21a whitenoise')
    #mdl = fit_fixedlstm(list(range(4)), ['naturalscene_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-11-21a', description='fixedlstm naturalscene on 3154c9 convnet activities')
    #mdl = fit_fixedlstm(list(range(19)), ['naturalscene_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-11-21b', description='fixedlstm naturalscene on 15-11-21b')

    # Run conv-conv-architectures with much higher sigmas
    #mdl = fit_generalizedconvnet(gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=6000, description='noise injection of 15std', sigma=15.0)

    # Run fixed_rnns
    #mdl = fit_fixedrnn(list(range(4)), ['whitenoise_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-11-21a', description='fixedrnn whitenoise on 15-11-21a')
    #mdl = fit_fixedrnn(list(range(19)), ['whitenoise_affine'], ['whitenoise_affine', 'naturalscene_affine'], '15-11-21b', description='fixedrnn whitenoise on 15-11-21b', num_affine=32)

    #mdl = fit_generalizedconvnet(gc_15_11_21a, ['naturalscene'], ['whitenoise', 'naturalscene'], '15-11-21a', nclip=6000, description='conv-affine-affine on 15-11-21a naturalscene with activity reg, parametric softplus, and injected noise')
    mdl = fit_generalizedconvnet(list(range(7)), ['tpinknoise'], ['tpinknoise', 'spinknoise'], '16-05-17', nclip=0, description='conv-conv-affine on Jeffs 16-05-17 temporal pink noise experiment')
    mdl = fit_generalizedconvnet(list(range(7)), ['spinknoise'], ['tpinknoise', 'spinknoise'], '16-05-17', nclip=0, description='conv-conv-affine on Jeffs 16-05-17 spatial pink noise experiment')
