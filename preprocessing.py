"""
Preprocessing utility functions for loading and formatting experimental data

"""

from __future__ import absolute_import
import numpy as np
import os
import h5py
from scipy.stats import zscore
from utils import rolling_window, notify, Batch

__all__ = ['datagen', 'loadexpt']

# custom data directories for different machines based on os.uname
datadirs = {
    'mbp': os.path.expanduser('~/experiments/data/'),
    'lenna': os.path.expanduser('~/experiments/data/'),
    'lane.local': os.path.expanduser('~/Documents/Stanford/00 Baccus Lab/Data 2015_10_07/')
}


def loadexpt(cellidx, filename, method, history, fraction=1.):
    """
    Loads an experiment from disk

    Parameters
    ----------
    cellidx : int
        Index of the cell to load

    filename : string
        Name of the hdf5 file to load

    method : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz stimulus

    fraction : float, optional
        Fraction of the experiment to load, must be between 0 and 1. (Default: 1.0)

    """

    assert fraction > 0 and fraction <= 1, "Fraction of data to load must be between 0 and 1"

    # currently only works with the Oct. 07, 15 experiment
    expt = '15-10-07'

    with notify('Loading {}ing data'.format(method)):

        # load the hdf5 file
        f = h5py.File(os.path.join(datadirs[os.uname()[1]], expt, filename + '.h5'), 'r')

        # length of the experiment
        expt_length = f[method]['time'].size
        num_samples = int(np.floor(expt_length * fraction))

        # load the stimulus
        stim = zscore(np.array(f[method]['stimulus'][:num_samples]).astype('float32'))

        # reshaped stimulus (nsamples, time/channel, space, space)
        stim_reshaped = np.rollaxis(np.rollaxis(rolling_window(stim, history, axis=0), 2), 3, 1)

        # get the response for this cell
        resp = np.array(f[method]['response/firing_rate_10ms'][cellidx, history:num_samples])

    return Batch(stim_reshaped, resp)


def datagen(batchsize, X, y):
    """
    Returns a generator that yields batches of data for one pass through the data

    Parameters
    ----------
    batchsize : int

    X : array_like

    y : array_like

    """

    # number of samples
    nsamples = y.size

    # compute the number of batches per epoch
    num_batches = int(np.floor(float(nsamples) / batchsize))

    # reshuffle indices
    N = num_batches * batchsize
    indices = np.random.choice(N, N, replace=False).reshape(num_batches, batchsize)

    # for each batch in this epoch
    for inds in indices:

        # yield data
        yield X[inds, ...], y[inds]
