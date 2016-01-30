"""
Preprocessing utility functions for loading and formatting experimental data

"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import h5py
from scipy.stats import zscore
from .utils import notify, Batch

__all__ = ['loadexpt']


def loadexpt(expt, cells, filename, train_or_test, history, load_fraction=1.0):
    """Loads an experiment from an h5 file on disk

    Parameters
    ----------
    expt : str
        The date of the experiment to load in YY-MM-DD format (e.g. '15-10-07')

    cells : int or list of ints
        Indices of the cells to load from the experiment

    filename : string
        Name of the hdf5 file to load (e.g. 'whitenoise' or 'naturalscene')

    train_or_test : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz stimulus

    load_fraction : float, optional
        Fraction of the expt to load, must be between 0 and 1 (Default: 1.0)

    """

    assert load_fraction > 0 and load_fraction <= 1, "Fraction of data to load must be between 0 and 1"
    assert history > 0 and type(history) is int, "Temporal history parameter must be a positive integer"
    assert train_or_test in ('train', 'test'), "The train_or_test parameter must be 'train' or 'test'"

    with notify('Loading {}ing data'.format(train_or_test)):

        # load the hdf5 file
        filepath = os.path.join(os.path.expanduser('~/experiments/data'),
                                expt,
                                filename + '.h5')

        with h5py.File(filepath, mode='r') as f:

            # get the length of the experiment & the number of samples to load
            expt_length = f[train_or_test]['time'].size
            num_samples = int(np.floor(expt_length * load_fraction))

            # load the stimulus as a float32 array, and z-score it
            stim = zscore(np.array(f[train_or_test]['stimulus'][:num_samples]).astype('float32'))

            # reshape into the Toeplitz matrix (nsamples, history, *stim_dims)
            stim_reshaped = rolling_window(stim, history, time_axis=0)

            # get the response for this cell (nsamples, ncells)
            resp = np.array(f[train_or_test]['response/firing_rate_10ms'][cells, history:num_samples]).T

    return Batch(stim_reshaped, resp)


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

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

    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr
