"""
Helper utilities

"""

from __future__ import print_function
from contextlib import contextmanager
from scipy.stats import pearsonr
from os import mkdir, uname, getlogin
from os.path import join, expanduser
from time import strftime
from collections import namedtuple
import numpy as np
import sys
import csv

__all__ = ['notify', 'rolling_window', 'mksavedir', 'tocsv', 'save_markdown',
           'metric', 'Batch']


Batch = namedtuple('Batch', ['X', 'y'])


@contextmanager
def notify(title):
    """
    Context manager for printing messages of the form 'Loading... Done.'

    Parameters
    ----------
    title : string
        A message / title to print

    Usage
    -----
    with notify('Loading'):
        # do long running task
        time.sleep(0.5)
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
    userdir = uname()[1] + '.' + getlogin()
    savedir = join(expanduser(basedir), userdir, now)

    # create the directory
    mkdir(savedir)

    return savedir


def tocsv(filename, array, fmt=''):
    """
    Write the data in the given array to a CSV file

    """

    row = [('{' + fmt + '}').format(x) for x in array]

    # add .csv to the filename if necessary
    if not filename.endswith('.csv'):
        filename += '.csv'

    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(row)


def tomarkdown(filename, lines):
    """
    Write the given lines to a markdown file

    """

    # add .csv to the filename if necessary
    if not filename.endswith('.md'):
        filename += '.md'

    with open(filename, 'a') as f:
        f.write('\n'.join(lines))


def metric(yhat,yobs):
    """
    Metric for comparing predicted and observed firing rates

    """
    return pearsonr(yhat, yobs)[0]


def rolling_window(array, window, axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    axis : 'first' or 'last', optional
        The axis of the time dimension (default: 'first')

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size `window`.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
               [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
               [ 6.,  7.,  8.]])

    """

    # flip array dimensinos if the time axis is the first axis
    if axis == 0:
        array = array.T

    elif axis == -1:
        pass

    else:
        raise ValueError("Time axis must be 0 or -1")

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
