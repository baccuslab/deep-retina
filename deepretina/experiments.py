"""
Preprocessing utility functions for loading and formatting experimental data

"""

from __future__ import absolute_import, division, print_function
import os
from functools import partial
from itertools import repeat
from collections import namedtuple
import numpy as np
import h5py
from scipy.stats import zscore
from .utils import notify, allmetrics

Exptdata = namedtuple('Exptdata', ['X', 'y'])
__all__ = ['Experiment', 'loadexpt']


class Experiment(object):
    """Lightweight class to keep track of loaded experiment data"""

    def __init__(self, expt, cells, train_filenames, test_filenames, history, batchsize, holdout=0.1, dt=1e-2, load_fraction=1.0):
        """Keeps track of experimental data

        Parameters
        ----------
        expt : string
            The experiment date or name

        cells : list
            Which cells from this experiment to train on

        train_filenames : list of strings
            Which h5 file to load for training (e.g. 'whitenoise' or 'naturalscene')
            If a list of strings is given (e.g. ['whitenoise', 'naturalscene']),
            the two experiments are concatenated

        test_filenames : list of strings
            Which h5 file to load for testing (same as train_filenames)

        history : int
            Temporal history, in samples, for the rolling window (Toeplitz stimulus)

        batchsize : int
            How many samples to include in each training batch

        holdout : fraction
            How much data to holdout (fraction of batches)

        dt : float
            The sampling period (in seconds). (default: 0.01)

        load_fraction : float
            What fraction of the training stimulus to load (default: 1.0)

        """

        # store experiment variables (for saving later)
        self.info = {
            'date': expt,
            'cells': cells,
            'train_datasets': ' + '.join(train_filenames),
            'test_datasets': ' + '.join(test_filenames),
            'history': history,
            'batchsize': batchsize,
            'load_fraction': load_fraction
        }

        assert holdout > 0 and holdout < 1, "holdout must be between 0 and 1"
        self.batchsize = batchsize
        self.dt = dt

        # partially apply function arguments to the loadexpt function
        load_data = partial(loadexpt, expt, cells, history=history, load_fraction=load_fraction)

        # load training data, and generate the train/validation split, for each filename
        self._train_data = {}
        self._train_batches = list()
        self._validation_batches = list()
        for filename in train_filenames:

            # load the training experiment as an Exptdata tuple
            self._train_data[filename] = load_data(filename, 'train')

            # generate the train/validation split
            length = self._train_data[filename].X.shape[0]
            train, val = _train_val_split(length, self.batchsize, holdout)

            # append these train/validation batches to the master list
            self._train_batches.extend(zip(repeat(filename), train))
            self._validation_batches.extend(zip(repeat(filename), val))

        # load the data for each experiment, store as a list of Exptdata tuple
        self._test_data = {filename: load_data(filename, 'test') for filename in test_filenames}

        # save batches_per_epoch for calculating # epochs later
        self.batches_per_epoch = len(self._train_batches) * self.batchsize

    def train(self, shuffle):
        """Returns a generator that yields batches of *training* data

        Parameters
        ----------
        shuffle : boolean
            Whether or not to shuffle the time points before making batches
        """
        # generate an order in which to go through the batches
        indices = np.arange(len(self._train_batches))
        if shuffle:
            np.random.shuffle(indices)

        # yield training data, one batch at a time
        for ix in indices:
            expt, inds = self._train_batches[ix]
            yield self._train_data[expt].X[inds], self._train_data[expt].y[inds]

    def validate(self, modelrate, metrics):
        """Evaluates the model on the validation set

        Parameters
        ----------
        modelrate : function
            A function that takes a spatiotemporal stimulus and predicts a firing rate
        """
        # choose a random validation batch
        expt, inds = self._validation_batches[np.random.randint(len(self._validation_batches))]

        # load the stimulus and response on this batch
        X = self._train_data[expt].X[inds]
        r = self._train_data[expt].y[inds]

        # make predictions
        rhat = modelrate(X)

        # evaluate using the given metrics
        return allmetrics(r, rhat, metrics), r, rhat

    def test(self, modelrate, metrics):
        """Tests model predictions on the repeat stimuli

        Parameters
        ----------
        modelrate : function
            A function that takes a spatiotemporal stimulus and predicts a firing rate
        """
        avg_scores = {}
        all_scores = {}
        for fname, exptdata in self._test_data.items():

            # get data and model firing rates
            r = exptdata.y
            rhat = modelrate(exptdata.X)

            # evaluate
            avg_scores[fname], all_scores[fname] = allmetrics(r, rhat, metrics)

        return avg_scores, all_scores

    @property
    def ndim(self):
        """Returns the number of filter dimensions"""
        key, _ = self._train_batches[0]
        return self._train_data['whitenoise'].X.shape[1:]


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

    with notify('Loading {}ing data for {}/{}'.format(train_or_test, expt, filename)):

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

    return Exptdata(stim_reshaped, resp)


def deprecated_loadexpt(cellidx, filename, method, history, fraction=1., cutout=False, cutout_cell=0):
    """
    Loads an experiment from disk

    ..Warning This function is deprecated!!! Only use this with old weights files

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
            filename = filename + '_cutout_cell%02d' %(cutout_cell + 1)

        # load the hdf5 file
        filepath = os.path.join(os.path.expanduser('~/experiments/data'),
                                expt,
                                filename + '.h5')
        f = h5py.File(filepath, 'r')

        # length of the experiment
        expt_length = f[method]['time'].size
        num_samples = int(np.floor(expt_length * fraction))

        # load the stimulus
        stim = zscore(np.array(f[method]['stimulus'][:num_samples]).astype('float32'))

        # reshaped stimulus (nsamples, time/channel, space, space)
        if history == 0:
            # don't create the toeplitz matrix
            stim_reshaped = stim
        else:
            stim_reshaped = np.rollaxis(np.rollaxis(rolling_window(stim, history, time_axis=0), 2), 3, 1)

        # get the response for this cell
        resp = np.array(f[method]['response/firing_rate_10ms'][cellidx, history:num_samples]).T

    return Exptdata(stim_reshaped, resp)


def _loadexpt_h5(expt, filename):
    """Loads an h5py reference to an experiment on disk"""

    filepath = os.path.join(os.path.expanduser('~/experiments/data'),
                            expt,
                            filename + '.h5')

    return h5py.File(filepath, mode='r')


def _train_val_split(length, batchsize, holdout):
    """Returns a set of training and a set of validation indices

    Parameters
    ----------
    length : int
        The total number of samples of data

    batchsize : int
        The number of samples to include in each batch

    holdout : float
        The fraction of batches to hold out for validation
    """
    # compute the number of available batches, given the fixed batch size
    num_batches = int(np.floor(length / batchsize))

    # the total number of samples for training
    total = int(num_batches * batchsize)

    # generate batch indices, and shuffle the deck of batches
    batch_indices = np.arange(total).reshape(num_batches, batchsize)
    np.random.shuffle(batch_indices)

    # compute the held out (validation) batches
    num_holdout = int(np.round(holdout * num_batches))

    return batch_indices[num_holdout:].copy(), batch_indices[:num_holdout].copy()


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


def deprecated_rolling_window(array, window, time_axis=0):
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
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
