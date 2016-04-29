"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, fixedlstm, experimentallstm
from deepretina.core import train
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper


@main_wrapper
def fit_experimentallstm(cells, timesteps, train_stimuli, test_stimuli, exptdate, readme=None):
    """Main script for fitting the experimentallstm

    author: Aran Nayebi
    """

    stim_shape = (timesteps, 40, 50, 50)
    ncells = len(cells)
    batchsize = 5000 #might want to make this smaller

    # get the compiled graph model
    model = experimentallstm(stim_shape, ncells)

    # compile the keras model
#    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

    for key in data._train_data.keys():
        data._train_data[key] = Exp
    data._train_data
    data._test_data['whitenoise']

    # create a monitor to track progress
    monitor = KerasMonitor('experimentallstm', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model

@main_wrapper
def fit_fixedlstm(cells, train_stimuli, test_stimuli, exptdate, timesteps, l2_reg, readme=None):
    """Main script for fitting a fixedlstm

    author: Aran Nayebi
    """
    print(timesteps)
    input_shape = (timesteps, 64)
    ncells = len(cells)
    batchsize = 100

    # get the fixedlstm layers
    layers = fixedlstm(input_shape, ncells, l2_reg=l2_reg)

    # compile the keras model
    model = sequential(layers, 'adam', loss='sub_poisson_loss')

    # load experiment data
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, input_shape[0], batchsize)

    # create a monitor to track progress
    monitor = KerasMonitor('fixedlstm', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model

if __name__ == '__main__':
#	reg = 0.01
#	p1 = 0.0
#	p2 = 0.0
	#mdl = fit_convnet([0, 1, 2, 3, 4], ['whitenoise'], ['whitenoise'], '15-10-07', reg, p1, p2,  description='WN convnet, l2_reg={}, dropout1={}, dropout2={}'.format(reg, p1, p2))

#    reg = 0.01
#    p1 = 0.25
#    p2 = 0.0
#    medOFF = [5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 36, 37]
#    slowOFF = [7, 19, 20, 21, 31, 34]
#    fastOFF = [1, 2, 3, 4]
#    medOFF = [x - 1 for x in medOFF]
#    slowOFF = [x - 1 for x in slowOFF]
#    fastOFF = [x - 1 for x in fastOFF] 
#    mdl = fit_convnet(medOFF, ['whitenoise'], ['whitenoise'], 'all-cells', reg, p1, p2,  description='WN convnet for mediumOFF cells, l2_reg={}, dropout1={}, dropout2={}'.format(reg, p1, p2))

#    mdl = fit_convnet(slowOFF, ['whitenoise'], ['whitenoise'], 'all-cells', reg, p1, p2,  description='WN convnet for slowOFF cells, l2_reg={}, dropout1={}, dropout2={}'.format(reg, p1, p2))

#    mdl = fit_convnet(fastOFF, ['whitenoise'], ['whitenoise'], 'all-cells', reg, p1, p2,  description='WN convnet for fastOFF cells, l2_reg={}, dropout1={}, dropout2={}'.format(reg, p1, p2))

#    p_arr1 = [0.75, 0.5, 0.25, 0]
#    p_arr2 = [0.10, 0.25, 0.5, 0.75, 0.85]
#    for p1 in p_arr1:
#        for p2 in p_arr2:
#            mdl = fit_convnet([0, 1, 2, 3, 4], ['whitenoise'], ['whitenoise'], '15-10-07', reg, p1, p2,  description='WN convnet, l2_reg={}, dropout1={}, dropout2={}'.format(reg, p1, p2))

#    mdl = fit_fixedlstm(list(range(37)), ['naturalscenes_affine'], ['whitenoise_affine', 'naturalscenes_affine'], 'all-cells', 1000, 0.01)
#    reg_arr = [0, 0.0001, 0.001, 0.01, 0.1]
#    p_arr = [0, 0.25, 0.5, 0.75]
#    for reg in reg_arr:
#        for p in p_arr:
	mdl = fit_experimentallstm([0, 1, 2, 3, 4], ['naturalscene'], ['naturalscene'], '15-10-07')
