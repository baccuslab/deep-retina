"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, generalizedconvnet, fixedlstm, conv_rgcs, functional, bn_cnn, bn_cnn_requ, cnn_bn_requ
from deepretina.core import train
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from deepretina.activations import ParametricSoftplus, ReQU
import tensorflow as tf


@main_wrapper
def fit_fixedlstm(cells, train_stimuli, test_stimuli, exptdate, hidden_units, readme=None):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    ncells = len(cells)
    input_shape = (1000,8*30*30)
    #input_shape = (1000,ncells)
    batchsize = 100


    with tf.device('/gpu:1'):
        # get the convnet layers
        layers = fixedlstm(input_shape, len(cells), num_hidden=hidden_units, weight_init='normal', l2_reg=0.01)

        # compile the keras model
        opt = Adam(lr=1e-3, decay=0.)
        model = sequential(layers, opt, loss='poisson')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize, nskip=0, zscore_flag=False)
    #data = Experiment(exptdate, cells, train_stimuli, test_stimuli, 1, batchsize, nskip=0, zscore_flag=False)
    # we have to do some tricky reshaping since we basically already have filter length (aka history)
    #for stim in train_stimuli:
    #    data._train_data[stim].X = np.squeeze(data._train_data[stim].X)

    #for stim in test_stimuli:
    #    data._test_data[stim].X = np.sqeeze(data._test_data[stim].X)

    # create a monitor to track progress
    monitor = KerasMonitor('fixedlstm', model, data, readme, save_every=20)
    #monitor = None

    with tf.device('/gpu:1'):
        # train
        train(model, data, monitor, num_epochs=100, shuffle=True)

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
    with tf.device('/gpu:1'):
        mdl = fit_fixedlstm(list(range(len(gc_15_11_21a))), ['naturalscene_flat_second_layer_7fc87c'], ['naturalscene_flat_second_layer_7fc87c'], '15-11-21a', 500, description='fixedlstm on conv activities of 7fc87c bn_cnn naturalscene')
