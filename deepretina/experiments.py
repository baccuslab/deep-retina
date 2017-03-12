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
dt = 1e-2
__all__ = ['Experiment', 'loadexpt']


class Experiment(object):
    """Class to keep track of loaded experiment data"""

    def __init__(self, expt, cells, train_filenames, test_filenames, history, batchsize, holdout=0.1, nskip=6000, zscore_flag=True):
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

        holdout : float
            How much data to holdout (fraction of batches) (must be between 0 and 1)

        nskip : int
            The number of stimulus frames to skip at the beginning of each stimulus block.
            Used to remove times when the retina is rapidly adapting to the change in stimulus
            statistics. (Default: 6000)

        zscore_flag : bool
            Whether stimulus should be zscored (default: True)
        """

        # store experiment variables (for saving later)
        self.info = {
            'date': expt,
            'cells': cells,
            'train_datasets': ' + '.join(train_filenames),
            'test_datasets': ' + '.join(test_filenames),
            'history': history,
            'batchsize': batchsize,
            'clipped': nskip * dt,
        }

        assert holdout >= 0 and holdout < 1, "holdout must be between 0 and 1"
        self.batchsize = batchsize
        self.dt = dt
        self.holdout = holdout

        # partially apply function arguments to the loadexpt function
        load_data = partial(loadexpt, expt, cells, history=history, zscore_flag=zscore_flag)

        # load training data, and generate the train/validation split, for each filename
        self._train_data = {}
        self._train_batches = list()
        self._validation_batches = list()
        for filename in train_filenames:

            # load the training experiment as an Exptdata tuple
            self._train_data[filename] = load_data(filename, 'train', nskip=nskip)

            # generate the train/validation split
            length = self._train_data[filename].X.shape[0]
            train, val = _train_val_split(length, self.batchsize, holdout)

            # append these train/validation batches to the master list
            self._train_batches.extend(zip(repeat(filename), train))
            self._validation_batches.extend(zip(repeat(filename), val))

        # load the data for each experiment, store as a list of Exptdata tuple
        self._test_data = {filename: load_data(filename, 'test', nskip=0) for filename in test_filenames}

        # save batches_per_epoch for calculating # epochs later
        self.batches_per_epoch = len(self._train_batches)

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
        rhat = modelrate({'stim': X})

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
            rhat = modelrate({'stim': exptdata.X})

            # evaluate
            avg_scores[fname], all_scores[fname] = allmetrics(r, rhat, metrics)

        return avg_scores, all_scores

    def cutout(self, xi, yi):
        """Cuts out the given slice from the stimuli in this experiment"""
        for stimset in ('_train_data', '_test_data'):
            stim = self.__dict__[stimset]
            for key, ex in stim.items():
                stim[key] = Exptdata(ex.X[:, :, xi, yi], ex.y)


def loadexpt(expt, cells, filename, train_or_test, history, nskip, zscore_flag=True):
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

    nskip : float, optional
        Number of samples to skip at the beginning of each repeat (Default: 0)

    zscore_flag : bool
        Whether to zscore the stimulus (Default: True)
    """
    assert history > 0 and type(history) is int, "Temporal history must be a positive integer"
    assert train_or_test in ('train', 'test'), "train_or_test must be 'train' or 'test'"

    with notify('Loading {}ing data for {}/{}'.format(train_or_test, expt, filename)):

        # load the hdf5 file
        filepath = os.path.join(os.path.expanduser('~/experiments/data'), expt, filename + '.h5')
        with h5py.File(filepath, mode='r') as f:

            expt_length = f[train_or_test]['time'].size

            # load the stimulus into memory as a numpy array
            stim = np.array(f[train_or_test]['stimulus']).astype('float32')

            # z-score the stimulus if desired
            if zscore_flag:
                stim = zscore(stim)

            # apply clipping to remove the stimulus just after transitions
            num_blocks = NUM_BLOCKS[expt] if train_or_test == 'train' else 1
            valid_indices = np.arange(expt_length).reshape(num_blocks, -1)[:, nskip:].ravel()

            # reshape into the Toeplitz matrix (nsamples, history, *stim_dims)
            stim_reshaped = rolling_window(stim[valid_indices], history, time_axis=0)

            # get the response for this cell (nsamples, ncells)
            resp = np.array(f[train_or_test]['response/firing_rate_10ms'][cells]).T[valid_indices]
            resp = resp[history:]

    return Exptdata(stim_reshaped, resp)


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
