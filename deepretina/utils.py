"""
Generic utilities for use in deepretina

"""

from __future__ import absolute_import, division, print_function
import sys
from contextlib import contextmanager
from . import metrics

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
        avg, cells = getattr(metrics, function)(r.T, rhat.T)
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
