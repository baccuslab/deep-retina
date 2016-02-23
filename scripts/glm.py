"""
Fits GLMs

"""

from limo import PoissonGLM, Feature
from deepretina.experiments import loadexpt, rolling_window, _loadexpt_h5
from deepretina.io import main_wrapper, MonitorGLM
import numpy as np


@main_wrapper
def fit_glm(cell, cells, filename, exptdate, readme=None):

    # cell and cell index
    cellindex = list(cells).index(cell)

    stimhist = 40
    ratehist = 51
    batchsize = 5000
    # algorithm = 'adam'
    algorithm = 'nag'

    # l2 regularization and learning rates for each set of parameters
    l2_stim_params = 2.5
    lr_stim = 1e-4
    l2_hist_params = 1.0
    lr_hist = lr_stim

    # load experiment data
    train = loadexpt(exptdate, cells, filename, 'train', stimhist)
    test = loadexpt(exptdate, cells, filename, 'test', stimhist)
    cutoff = np.max([stimhist, ratehist])
    testdata = [test.X[cutoff:, ...], rolling_window(test.y, ratehist)[:, :-1, :]]
    r_test = test.y[cutoff:, cellindex]

    # initialize proportional to the STA
    theta_init = 1e-4 * np.array(_loadexpt_h5(exptdate, 'whitenoise')['stas']['cell{:02}'.format(cell+1)])

    # build GLM features
    f_stim = Feature(train.X, l2=l2_stim_params, learning_rate=lr_stim, algorithm=algorithm)
    # f_stim.theta = theta_init.copy()
    f_hist = Feature(rolling_window(train.y, ratehist)[:, :-1, :], l2=l2_hist_params, learning_rate=lr_hist, algorithm=algorithm)

    # build the GLM
    glm = PoissonGLM([f_stim, f_hist], train.y[:, cellindex], dt=1e-2, batch_size=batchsize)

    # create a monitor
    monitor = MonitorGLM(glm, exptdate, filename, cell, testdata, r_test, readme, save_every=10)

    # train
    glm.fit(5, monitor)

    return glm


if __name__ == "__main__":

    goodcells = {
        '15-10-07': [0, 1, 2, 3, 4, 5],
        '15-11-21a': [6, 10, 12, 13],
        '15-11-21b': [0, 1, 3, 4, 5, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }

    # for testing
    # fit_glm(2, goodcells['15-10-07'], 'whitenoise', '15-10-07', description='GLM with nag')
     
    for exptdate, cells in goodcells.items():
        for ci in cells:
            fit_glm(ci, cells, 'whitenoise', exptdate, description='{}, Cell {}, whitenoise (v3)'.format(exptdate, ci))
            fit_glm(ci, cells, 'naturalscene', exptdate, description='{}, Cell {}, naturalscene (v3)'.format(exptdate, ci))
