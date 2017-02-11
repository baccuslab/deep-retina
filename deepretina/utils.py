"""
Generic utilities
"""

from __future__ import absolute_import, division, print_function
import sys
from contextlib import contextmanager
from . import metrics
import numpy as np
from scipy.stats import zscore
from itertools import combinations, repeat
from numbers import Number

__all__ = ['notify', 'allmetrics']


def allmetrics(r, rhat, functions):
    """Evaluates the given responses on all of the given metrics

    Parameters
    ----------
    r : array_like
        True response, with shape (# of samples, # of cells)

    rhat : array_like
        Model response, with shape (# of samples, # of cells)

    functions : list of strings
        Which functions from the metrics module to evaluate on
    """
    avg_scores = {}
    all_scores = {}
    for function in functions:
        avg, cells = getattr(metrics, function)(r.T, rhat['loss'].T)
        avg_scores[function] = avg
        all_scores[function] = cells

    return avg_scores, all_scores


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


def xcorr(x, y, maxlag, normalize=True):
    """Computes the cross correlation between two signals

    Parameters
    ----------
    x : array_like
        The first signal to correlate, must be 1-D

    y : array_like
        The second signal to correlate, must have the same shape as x

    maxlag : int
        The maximum lag length (in samples), must be a positive integer

    normalize : boolean, optional
        Whether or not to zscore the arrays before computing the lags,
        this forces the correlation to be between -1 and 1. (default: True)

    Returns
    -------
    lags : array_like
        An array of lag indices, ranging from -maxlag to maxlag

    corr : array_like
        The correlations of the two signals at each of the lags
    """
    assert type(maxlag) is int and maxlag > 0, \
        "maxlag must be a positive integer"

    assert x.shape == y.shape, \
        "The two arrays must have the same shape"

    if normalize:
        x = zscore(x.copy())
        y = zscore(y.copy())

    lags = np.arange(-maxlag, maxlag + 1)
    corr = np.zeros(len(lags))
    length = x.size

    for idx, lag in enumerate(lags):
        total = float(length - np.abs(lag))
        if lag < 0:
            corr[idx] = np.dot(x[:lag], y[-lag:]) / total
        elif lag > 0:
            corr[idx] = np.dot(x[lag:], y[:-lag]) / total
        else:
            corr[idx] = np.dot(x, y) / total

    return lags, corr


def pairs(n):
    """Return an iterator over n choose 2 possible unique pairs

    Usage
    -----
    >>> list(pairs(3))
    [(0, 1), (0, 2), (1, 2)]
    """
    return combinations(range(n), 2)


def tuplify(x, n):
    """Converts a number into a tuple with that number repeating

    Usage
    -----
    >>> tuplify(3, 5)
    (3, 3, 3, 3, 3)
    >>> tuplify((1,2), 5)
    (1, 2)
    """
    if isinstance(x, Number):
        x = tuple(repeat(x, n))
    return x


def cutout_indices(center, size=7, ndim=50):
    """Cuts out a region with the given size around a point"""
    xinds = slice(int(np.clip(center[0] - size, 0, ndim)), int(np.clip(center[0] + size + 1, 0, ndim)))
    yinds = slice(int(np.clip(center[1] - size, 0, ndim)), int(np.clip(center[1] + size + 1, 0, ndim)))
    return xinds, yinds


def _deprecated_cutout_indices(center, size=7, ndim=50):
    """Cuts out a region with the given size around a point"""
    xinds = slice(center[0] - size, center[0] + size)
    yinds = slice(center[1] - size, center[1] + size)
    return xinds, yinds
