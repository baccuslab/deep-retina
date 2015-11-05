from __future__ import absolute_import
import numpy as np
import os
import h5py
from scipy.stats import zscore

__all__ = ['datagen']


def datagen(cellidx, batchsize, expt='15-10-07', filename='naturalscene', method='train', history=40):
    """
    Returns a generator that yields batches of data

    Parameters
    ----------

    cellidx : int
        Which cell to train on

    batchsize : int
        How many samples to include in each batch

    expt : string, optional
        The experiment to load (currently only works with 15-10-07)

    filename : string, optional
        The name of the file to load ('naturalscene' or 'whitenoise')

    method : string, optional
        Either 'train' or 'test' for the class/type of data to load

    history : int, optional
        How many points to include in the filter history

    """

    # currently only works with the Oct. 07, 15 experiment
    if expt == '15-10-07':
        pass
    else:
        raise NotImplementedError('Did not recognize experiment: ' + str(expt))

    # custom data directories for different machines based on OS hostname
    datadirs = {
        'mbp': os.path.expanduser('~/experiments/data/'),
        'lenna': os.path.expanduser('~/experiments/data/')
    }

    # load the hdf5 file
    f = h5py.File(os.path.join(datadirs[os.uname()[1]], expt, filename + '.h5'), 'r')

    # load the stimulus
    stim = np.array(f[method]['stimulus']).astype('float32')

    # reshaped stimulus (nsamples, time/channel, space, space)
    stim_reshaped = np.rollaxis(np.rollaxis(rolling_window(stim, history, axis=0), 2), 3, 1)

    # get the response for this cell
    resp = np.array(f[method]['response/firing_rate_10ms'][cellidx, history:])

    # keep looping over epochs
    num_batches = int(np.floor(float(resp.size) / batchsize))
    while True:

        # reshuffle indices for this new epoch
        indices = np.arange(resp.size)
        np.random.shuffle(indices)
        indices = indices[:(batchsize * num_batches)].reshape(num_batches, batchsize)

        for batch_idx in range(num_batches):

            # select random indices for this batch
            inds = indices[batch_idx]

            # yield data
            yield zscore(stim_reshaped[inds, ...]), resp[inds]


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
