"""
Preprocessing utility functions for loading and formatting experimental data
"""
from collections import namedtuple

import h5py
import numpy as np
from os.path import join, expanduser
from scipy.stats import zscore
from deepretina import config
from random import shuffle
import tensorflow as tf
Sequence = tf.keras.utils.Sequence
import pyret.filtertools as ft

from .utils import notify

NUM_BLOCKS = {
    '15-10-07': 6,
    '15-11-21a': 6,
    '15-11-21b': 6,
    '16-01-07': 3,
    '16-01-08': 3,
}
CELLS = {
    '15-10-07': [0, 1, 2, 3, 4],
    '15-11-21a': [6, 10, 12, 13],
    '15-11-21b': [0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25],
    '16-01-07': [0, 2, 7, 10, 11, 12, 31],
    '16-01-08': [0, 3, 7, 9, 11],
    '16-05-31': [2, 3, 4, 14, 16, 18, 20, 25, 27]
}

Exptdata = namedtuple('Exptdata', ['X', 'y'])
__all__ = ['loadexpt', 'stimcut', 'CELLS']


class ExptSequence(Sequence):
    """Takes a list of ExptData and create a Sequence for training."""
    def __init__(self, experiments, train_or_valid="TRAIN", batch_size=1000,val_split=0.95):
        indices = []
        self.experiments = experiments
        self.batch_size = batch_size
        self.train_or_valid = train_or_valid
        self.experiment_n, self.experiment_ntrain = ntrain_by_experiment(experiments,val_split)

        nbatches = sum([np.ceil(len(e)/batch_size) for e in experiments])
        for i, experiment in enumerate(experiments):
            n = self.experiment_n[i]
            ntrain = self.experiment_ntrain[i]

            if train_or_valid=="TRAIN":
                bi = np.arange(0,ntrain-1,batch_size)
            else:
                bi = np.arange(ntrain,n-1,batch_size)

            # interpolate the experiments evenly
            for ranking, batch_start in zip(np.linspace(0,nbatches,len(bi)), bi):
                if self.train_or_valid=="TRAIN":
                    batch_end = min(ntrain, batch_start+self.batch_size)
                else:
                    batch_end = min(n, batch_start+self.batch_size)
                indices.append((ranking, i, batch_start, batch_end))

        self.indices = [x[1:] for x in sorted(indices,key=lambda y: y[0])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        expt, batch_start, batch_end = self.indices[idx]
        experiment = self.experiments[expt]

        return (experiment.X[batch_start:batch_end],
               experiment.y[batch_start:batch_end])

def loadexpt(expt, cells, filename, train_or_test, history, nskip, cutout_width=None):
    """Loads an experiment from an h5 file on disk

    Parameters
    ----------
    expt : str
        The date of the experiment to load in YY-MM-DD format (e.g. '15-10-07')

    cells : list of ints
        Indices of the cells to load from the experiment

    filename : string
        Name of the hdf5 file to load (e.g. 'whitenoise' or 'naturalscene')

    train_or_test : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz stimulus

    nskip : float, optional
        Number of samples to skip at the beginning of each repeat

    cutout_width : int, optional
        If not None, cuts out the stimulus around the STA (assumes cells is a scalar)
    """
    assert history > 0 and type(history) is int, "Temporal history must be a positive integer"
    assert train_or_test in ('train', 'test'), "train_or_test must be 'train' or 'test'"

    with notify('Loading {}ing data for {}/{}'.format(train_or_test, expt, filename)):

        # get whitenoise STA for cutout stimulus
        if cutout_width is not None:
            assert len(cells) == 1, "cutout must be used with single cells"
            wn = _loadexpt_h5(expt, 'whitenoise')
            sta = np.array(wn[f'train/stas/cell{cells[0]+1:02d}']).copy()
            py, px = ft.filterpeak(sta)[1]

        # load the hdf5 file
        with _loadexpt_h5(expt, filename) as f:

            expt_length = f[train_or_test]['time'].size

            # load the stimulus into memory as a numpy array, and z-score it
            if cutout_width is None:
                stim = zscore(np.array(f[train_or_test]['stimulus']).astype('float32'))
            else:
                stim = zscore(ft.cutout(np.array(f[train_or_test]['stimulus']), idx=(px, py), width=cutout_width).astype('float32'))

            # TODO I have no idea what num_blocks is for...is this reasonable?
            if expt in NUM_BLOCKS:
                # apply clipping to remove the stimulus just after transitions
                num_blocks = NUM_BLOCKS[expt] if train_or_test == 'train' and nskip > 0 else 1
            else:
                num_blocks = 1
            valid_indices = np.arange(expt_length).reshape(num_blocks, -1)[:, nskip:].ravel()

            # reshape into the Toeplitz matrix (nsamples, history, *stim_dims)
            stim_reshaped = rolling_window(stim[valid_indices], history, time_axis=0)

            # get the response for this cell (nsamples, ncells)
            if cells=="all":
                resp = np.array(f[train_or_test]['response/firing_rate_10ms']).T[valid_indices]
            else:
                resp = np.array(f[train_or_test]['response/firing_rate_10ms'][cells]).T[valid_indices]
            resp = resp[history:]

    return Exptdata(stim_reshaped, resp)


def _loadexpt_h5(expt, filename):
    """Loads an h5py reference to an experiment on disk"""
    filepath = join(expanduser(config.data_dir), expt, filename + '.h5')
    return h5py.File(filepath, mode='r')

def load_multiple_expt(list_expt_cells_filename, train_or_test, history, nskip, cutout_width=None):
    """Load multiple experiments into a single array. Reshape for G_CONV.

    Parameters
    ----------
    list_expt_cells_filename : List(Tuple(expt, cells, filename))

    Others are the same as load_ext
    """
    expts = []
    nsamples = 0
    input_shape = None
    for item in list_expt_cells_filename:
        expt, cells, filename = item
        data = loadexpt(expt, cells, filename, train_or_test, history, nskip, cutout_width)
        y = np.reshape(data.y,[*data.y.shape,1,1])
        expdata = Exptdata(data.X,y)
        expts.append(expdata)
        nsamples+= len(expdata.X)
        input_shape = expdata.X.shape[1:]

    return expts, input_shape, nsamples

def ntrain_by_experiment(experiments,val_split=0.95):
    experiment_ntrain = []
    experiment_n = []

    for i, experiment in enumerate(experiments):
        n = len(experiment.X)
        experiment_n.append(n)
        ntrain = int(np.floor(n*0.95))
        experiment_ntrain.append(ntrain)

    return np.array(experiment_n), np.array(experiment_ntrain)

def stimcut(data, expt, ci, width=11):
    """Cuts out a stimulus around the whitenoise receptive field"""

    # get the white noise STA for this cell
    wn = _loadexpt_h5(expt, 'whitenoise')
    sta = np.array(wn[f'train/stas/cell{ci+1:02d}']).copy()

    # find the peak of the sta
    xc, yc = ft.filterpeak(sta)[1]

    # cutout stimulus
    X, y = data
    Xc = ft.cutout(X, idx=(yc, xc), width=width)
    yc = y[:, ci].reshape(-1, 1)
    return Exptdata(Xc, yc)


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
