"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, train
from deepretina.experiments import Experiment
from deepretina.io import Monitor
import inspect
import subprocess
from functools import wraps


def main_wrapper(func):
    @wraps(func)
    def mainscript(*args, **kwargs):

        # get information about this function call
        func.__name__
        source = inspect.getsource(func)
        commit = subprocess.check_output(["git", "describe", "--always"])

        # build a markdown string containing this information
        kwargs['readme'] = '\n'.join(['# deep-retina script',
                                      '### git commit',
                                      commit,
                                      '### function call',
                                      '{}({}, {})'.format(func.__name__, args, kwargs),
                                      '### source',
                                      '```python',
                                      source,
                                      '```'])

        func(*args, **kwargs)

    return mainscript


@main_wrapper
def fit_convnet(cells, stimulus, exptdate='15-10-07', readme=None):
    """Demo code for fitting a convnet model"""

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 500

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(2, 4),
                     filter_size=(3, 3), weight_init='normal', l2_reg=0.01)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    data = Experiment(exptdate, cells, stimulus, stim_shape[0], batchsize, load_fraction=0.1)

    # create a monitor to track progress
    monitor = Monitor('convnet', model, data, readme, save_every=5)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':
    mdl = fit_convnet([0, 1, 2, 3, 4], 'naturalscene')
