"""
Fits GLMs

"""

from limo import PoissonGLM, Feature
from deepretina.experiments import Experiment
from deepretina.io import main_wrapper, Monitor
from pyret.stimulustools import rolling_window


@main_wrapper
def fit_glm(ci, cells, train_stimulus, exptdate, readme=None):

    stimhist = 40
    ratehist = 50
    batchsize = 5000

    # load experiment data
    test_stimuli = ['whitenoise', 'naturalscene']
    data = Experiment(exptdate, cells, [train_stimulus], test_stimuli, stimhist, batchsize)
    exptdata = data._train_data[train_stimulus]

    # build GLM features
    f_stim = Feature(exptdata.X)
    f_hist = Feature(rolling_window(exptdata.y, ratehist + 1)[:, :-1, :])

    # build the GLM
    glm = PoissonGLM([f_stim, f_hist], exptdata.y[:, ci], 1e-2, batch_size=5000)

    # create a monitor
    monitor = Monitor('GLM', glm, data, readme, save_every=10)

    # train
    glm.fit(100, monitor)

    return glm


if __name__ == "__main__":
    fit_glm()
