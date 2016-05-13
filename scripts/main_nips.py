"""
NIPS main script
"""

from __future__ import absolute_import
from deepretina.models import sequential, nips_conv
from deepretina.core import train
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper


@main_wrapper
def fit_nips_conv(cells, train_stimuli, exptdate, readme=None):
    """Main script for fitting a multilayered convnet"""
    stim_shape = (40, 50, 50)
    batchsize = 5000

    # get the convnet layers
    layers = nips_conv(len(cells))

    # compile the keras model
    model = sequential(layers, 'adam', loss='poisson')

    # load experiment data
    test_stimuli = ['whitenoise', 'naturalscene']
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize, nskip=6000)

    # create a monitor to track progress
    monitor = KerasMonitor('multilayered_convnet', model, data, readme, save_every=20)

    # train
    train(model, data, monitor, num_epochs=50)

    return model


if __name__ == '__main__':

    # ===============
    # 15-10-07 (Aran)
    # ===============
    # cells = [0, 1, 2, 3, 4]
    # fit_nips_conv(cells, ['naturalscene'], '15-10-07', description="15-10-07 nips model on naturalscene")
    # fit_nips_conv(cells, ['whitenoise'], '15-10-07', description="15-10-07 nips model on whitenoise")

    # ================
    # 15-11-21a (Lane)
    # ================
    cells = [6, 10, 12, 13]
    fit_nips_conv(cells, ['naturalscene'], '15-11-21a', description="15-11-21a nips model on naturalscene")
    fit_nips_conv(cells, ['whitenoise'], '15-11-21a', description="15-11-21a nips model on whitenoise")

    # ================
    # 15-11-21b (Niru)
    # ================
    # cells = [0, 1, 3, 4, 5, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    # fit_nips_conv(cells, ['naturalscene'], '15-11-21b', description="15-11-21b nips model on naturalscene")
    # fit_nips_conv(cells, ['whitenoise'], '15-11-21b', description="15-11-21b nips model on whitenoise")
