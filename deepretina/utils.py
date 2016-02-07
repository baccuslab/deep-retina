"""
Generic utilities for use in deepretina

"""

from __future__ import absolute_import, division, print_function
import sys
from collections import namedtuple
from contextlib import contextmanager
import numpy as np

__all__ = ['batchify', 'Batch', 'notify']

Batch = namedtuple('Batch', ['X', 'y'])


def batchify(batchsize, X, y, shuffle):
    """Returns a generator that yields batches of data (one epoch)

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
