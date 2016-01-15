"""
Preprocessing utility functions for loading and formatting experimental data

"""

from __future__ import absolute_import
import numpy as np
import os
import h5py
from scipy.stats import zscore
from scipy.special import gamma
from .utils import rolling_window, notify, Batch

__all__ = ['datagen', 'loadexpt']

# custom data directories for different machines based on os.uname
datadirs = {
    'mbp': os.path.expanduser('~/experiments/data/'),
    'lenna': os.path.expanduser('~/experiments/data/'),
    'lane.local': os.path.expanduser('~/experiments/data/')
}


def loadexpt(cellidx, filename, method, history, fraction=1., mean_adapt=False,
    cutout=False, cutout_cell=0):
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

        # select different filename if you want a cutout
        if cutout:
            filename = filename + '_cutout_cell%02d' %(cutout_cell+1)

        # load the hdf5 file
        f = h5py.File(os.path.join(datadirs[os.uname()[1]], expt, filename + '.h5'), 'r')

        # length of the experiment
        expt_length = f[method]['time'].size
        num_samples = int(np.floor(expt_length * fraction))

        # load the stimulus
        stim = zscore(np.array(f[method]['stimulus'][:num_samples]).astype('float32'))

        # photoreceptor model of mean adaptation
        if mean_adapt:
            stim = pr_filter(10e-3, stim)

        # reshaped stimulus (nsamples, time/channel, space, space)
        if history == 0:
            # don't create the toeplitz matrix
            stim_reshaped = stim
        else:
            stim_reshaped = np.rollaxis(np.rollaxis(rolling_window(stim, history, time_axis=0), 2), 3, 1)

        # get the response for this cell
        resp = np.array(f[method]['response/firing_rate_10ms'][cellidx, history:num_samples]).T

    return Batch(stim_reshaped, resp)


def datagen(batchsize, X, y, shuffle=True):
    """
    Returns a generator that yields batches of data for one pass through the data

    Parameters
    ----------
    batchsize : int

    X : array_like

    y : array_like

    """

    # total number of samples
    training_data_maxlength = y.shape[0]

    # compute the number of available batches of a fixed size
    num_batches = int(np.floor(float(training_data_maxlength) / batchsize))

    # number of samples we are going to used
    N = int(num_batches * batchsize)

    # generate indices
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    # reshape into batches
    indices = indices.reshape(num_batches, batchsize)

    # for each batch in this epoch
    for inds in indices:

        # yield data
        yield X[inds, ...], y[inds]


def pr_filter(dt, stim, tau_y=0.033, ny=4., tau_z=0.019, nz=10., alpha=1., beta=0.16, eta=0.23):
    """
    Filter the given stimulus using a model of photoreceptor adaptation

    Dynamical Adaptation in Photoreceptors (Clark et. al., 2015)
    http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289

    """

    # build the two filters
    t = np.arange(dt, 0.5, dt)
    Ky = dt * _make_filter(t, tau_y, ny)
    Kz = eta * Ky + (1 - eta) * dt * _make_filter(t, tau_z, nz)

    # filter the stimulus
    y = np.zeros_like(stim)
    z = np.zeros_like(stim)
    T = stim.shape[0]
    for row in range(stim.shape[1]):
        for col in range(stim.shape[2]):
            y[:, row, col] = np.convolve(stim[:,row,col], Ky, mode='full')[:T]
            z[:, row, col] = np.convolve(stim[:,row,col], Kz, mode='full')[:T]

    # return the filtered stimulus
    return (alpha * y) / (1 + beta * z)


def _make_filter(t, tau, n):
    return ((t ** n) / (gamma(n+1) * tau ** (n + 1))) * np.exp(-t / tau)
