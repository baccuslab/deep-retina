"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, generalizedconvnet, fixedlstm
from deepretina.core import train
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper
from deepretina.toolbox import load_model, modify_model
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.models import model_from_json
import os


@main_wrapper
def retrain(model_hash, cells, train_stimuli, test_stimuli, exptdate, readme=None, changed_params=None, weight_name='best_weights.h5', nclip=0):
    """Main script for fitting a convnet

    author: Lane McIntosh
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # load the keras model
    model_path = os.path.expanduser('~/deep-retina-results/database/%s/' %model_hash)
    if changed_params is not None:
        model = modify_model(model_path, weight_name, changed_params)
    else:
        model = load_model(model_path, weight_name)

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize, nskip=nclip)

    # create a monitor to track progress
    monitor = KerasMonitor('convnet', model, data, readme, save_every=20)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':
    gc_15_10_07 = [0,1,2,3,4]
    gc_15_11_21a = [6,10,12,13]
    gc_15_11_21b = [0,1,3,4,5,8,9,13,14,16,17,18,19,20,21,22,23,24,25]
    gc_16_01_07 = [0,2,7,10,11,12,31]
    gc_16_01_08 = [0,3,7,9,11]
    gc_all_cells = list(range(37))
    #mdl = retrain('3dd884 convnet', gc_15_10_07, ['whitenoise'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=5000)
    mdl = retrain('3520cd convnet', gc_15_10_07, ['naturalscene'], ['whitenoise', 'naturalscene'], '15-10-07', nclip=5000)
