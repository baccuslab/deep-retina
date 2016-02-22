"""
Fits GLMs

"""

from limo import PoissonGLM, Feature
from deepretina.experiments import loadexpt, rolling_window
from deepretina.io import main_wrapper, MonitorGLM
import numpy as np


@main_wrapper
def fit_glm(ci, cells, filename, exptdate, readme=None):

    stimhist = 40
    ratehist = 51
    batchsize = 5000

    # load experiment data
    train = loadexpt(exptdate, cells, filename, 'train', stimhist)
    test = loadexpt(exptdate, cells, filename, 'test', stimhist)
    cutoff = np.max([stimhist, ratehist])
    testdata = [test.X[cutoff:, ...], rolling_window(test.y, ratehist)]
    r_test = test.y[cutoff:, ci]

    # build GLM features
    f_stim = Feature(train.X)
    f_hist = Feature(rolling_window(train.y, ratehist)[:, :-1, :])

    # build the GLM
    glm = PoissonGLM([f_stim, f_hist], train.y[:, ci], dt=1e-2, batch_size=batchsize)

    # create a monitor
    monitor = MonitorGLM(glm, exptdate, filename, ci, testdata, r_test, readme, save_every=10)

    # train
    glm.fit(10, monitor)

    return glm


if __name__ == "__main__":
    fit_glm()
