"""
Helper utilities for saving models and model outputs

"""

from __future__ import print_function
from contextlib import contextmanager
from os import mkdir, uname, getenv
from os.path import join, expanduser
from time import strftime
from collections import namedtuple
from pyret.stimulustools import rolling_window
from scipy.stats import zscore
import numpy as np
import sys
import h5py

__all__ = ['notify', 'mksavedir', 'loadexpt', 'batchify', 'Batch']

Batch = namedtuple('Batch', ['X', 'y'])


def loadexpt(cellidx, filename, train_or_test, history, fraction=1., cutout=False, cutout_cell=0, exptdate='15-10-07'):
    """Loads an experiment from disk

    Parameters
    ----------
    cellidx : int
        Index of the cell to load

    filename : string
        Name of the hdf5 file to load

    train_or_test : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz stimulus

    fraction : float, optional
        Fraction of the experiment to load, must be between 0 and 1. (Default: 1.0)

    """

    assert fraction > 0 and fraction <= 1, "Fraction of data to load must be between 0 and 1"

    with notify('Loading {}ing data'.format(train_or_test)):

        # select different filename if you want a cutout
        if cutout:
            filename = filename + '_cutout_cell%02d' %(cutout_cell + 1)

        # load the hdf5 file
        f = h5py.File(join(expanduser('~/experiments/data'), exptdate, filename + '.h5'), 'r')

        # length of the experiment
        expt_length = f[train_or_test]['time'].size
        num_samples = int(np.floor(expt_length * fraction))

        # load the stimulus
        stim = zscore(np.array(f[train_or_test]['stimulus'][:num_samples]).astype('float32'))

        # reshaped stimulus (nsamples, time/channel, space, space)
        if history == 0:
            # don't create the toeplitz matrix
            stim_reshaped = stim
        else:
            stim_reshaped = np.rollaxis(np.rollaxis(rolling_window(stim, history, time_axis=0), stim.ndim - 1), stim.ndim, 1)

        # get the response for this cell
        resp = np.array(f[train_or_test]['response/firing_rate_10ms'][cellidx, history:num_samples]).T

    return Batch(stim_reshaped, resp)


def batchify(batchsize, X, y, shuffle):
    """Returns a generator that yields batches of data for one pass through the data

    Parameters
    ----------
    batchsize : int
        The number of samples to include in each batch

    X : array_like
        Array of stimuli, the first dimension indexes each sample

    y : array_like
        Array of responses, the first dimension indexes each sample

    shuffle : boolean
        If true, the samples are shuffled before being put into batches

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


@contextmanager
def notify(title):
    """Context manager for printing messages of the form 'Loading... Done.'

    Parameters
    ----------
    title : string
        A message / title to print

    Usage
    -----
    >>> with notify('Loading'):
    >>>    # do long running task
    >>>    time.sleep(0.5)
    >>> Loading... Done.

    """

    print(title + '... ', end='')
    sys.stdout.flush()
    try:
        yield
    finally:
        print('Done.')


def mksavedir(basedir='~/Dropbox/deep-retina/saved', prefix=''):
    """
    Makes a new directory for saving models

    Parameters
    ----------
    basedir : string, optional
        Base directory to store model results in

    prefix : string, optional
        Prefix to add to the folder (name of the model or some other identifier)

    """

    assert type(prefix) is str, "prefix must be a string"

    # get the current date and time
    now = strftime("%Y-%m-%d %H.%M.%S") + " " + prefix

    # the save directory is the given base directory plus the current date/time
    userdir = uname()[1] + '.' + getenv('USER')
    savedir = join(expanduser(basedir), userdir, now)

    # create the directory
    mkdir(savedir)

    return savedir


def tomarkdown(filename, lines):
    """
    Write the given lines to a markdown file

    """

    # add .md to the filename if necessary
    if not filename.endswith('.md'):
        filename += '.md'

    with open(filename, 'a') as f:
        f.write('\n'.join(lines))
