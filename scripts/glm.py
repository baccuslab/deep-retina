"""
Fits GLMs

"""

from limo import PoissonGLM, Feature
from deepretina.experiments import loadexpt, rolling_window, _loadexpt_h5
from deepretina.io import main_wrapper, MonitorGLM
import numpy as np


@main_wrapper
def fit_glm(cell, cells, filename, exptdate, lf=1.0, num_epochs=3, readme=None):

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
    train = loadexpt(exptdate, cells, filename, 'train', stimhist, load_fraction=lf)
    test = loadexpt(exptdate, cells, filename, 'test', stimhist)
    # cutoff = np.max([stimhist, ratehist])
    cutoff = 40
    testdata = [test.X[cutoff:, ...], rolling_window(test.y, ratehist)[:, :-1, :]]
    r_test = test.y[cutoff:, cellindex]

    # initialize proportional to the STA
    theta_init = 1e-4 * np.array(_loadexpt_h5(exptdate, 'whitenoise')['stas']['cell{:02}'.format(cell+1)])

    # build GLM features
    f_stim = Feature(train.X, l2=l2_stim_params, learning_rate=lr_stim, algorithm=algorithm)
    # f_stim.theta = theta_init.copy()
    # f_hist = Feature(rolling_window(train.y, ratehist)[:, :-1, :], l2=l2_hist_params, learning_rate=lr_hist, algorithm=algorithm)

    # build the GLM
    glm = PoissonGLM([f_stim], train.y[:, cellindex], dt=1e-2, batch_size=batchsize)

    # create a monitor
    monitor = MonitorGLM(glm, exptdate, filename, cell, testdata, r_test, readme, save_every=10)

    # train
    glm.fit(num_epochs, monitor)

    return glm


if __name__ == "__main__":

    goodcells = {
        '15-10-07': [0, 1, 2, 3, 4, 5, 6, 7],
        '15-11-21a': [0, 1, 6, 10, 12, 13],
        '15-11-21b': [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }

    # different amounts of data
    cells = [0, 1, 2, 3, 4]
    num_epochs = [96, 48, 21, 18, 12, 6]
    fractions = [0.01668, 0.03336, 0.06671, 0.13342, 0.26684, 0.53369]
    for ci in cells:
        for lf, nb in zip(fractions, num_epochs):
            fit_glm(ci, cells, 'whitenoise', '15-10-07', lf=lf, num_epochs=nb, description='GLM on cell {} with {} minutes of data, whitenoise, 15-10-07'.format(ci, 59.96 * lf))

    # for testing
    # fit_glm(2, goodcells['15-10-07'], 'whitenoise', '15-10-07', description='GLM with nag')

    # unfinished = [('15-10-07', [6, 7]), ('15-11-21a', [0, 1]), ('15-11-21b', [2, 10, 15])]

    # for exptdate, cells in unfinished:
        # for ci in cells:
            # fit_glm(ci, goodcells[exptdate], 'whitenoise', exptdate, description='{}, Cell {}, whitenoise (v3)'.format(exptdate, ci))
            # fit_glm(ci, goodcells[exptdate], 'naturalscene', exptdate, description='{}, Cell {}, whitenoise (v3)'.format(exptdate, ci))

    # for exptdate, cells in goodcells.items():
        # for ci in cells:
            # fit_glm(ci, cells, 'whitenoise', exptdate, description='{}, Cell {}, whitenoise (v3)'.format(exptdate, ci))
            # fit_glm(ci, cells, 'naturalscene', exptdate, description='{}, Cell {}, naturalscene (v3)'.format(exptdate, ci))
