"""
Generic utilities
"""
from __future__ import absolute_import, division, print_function

import sys
from contextlib import contextmanager
from itertools import repeat
from numbers import Number

import numpy as np

__all__ = ['notify']


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
    xinds = slice(int(np.clip(center[0] - size, 0, ndim)),
                  int(np.clip(center[0] + size + 1, 0, ndim)))
    yinds = slice(int(np.clip(center[1] - size, 0, ndim)),
                  int(np.clip(center[1] + size + 1, 0, ndim)))
    return xinds, yinds
