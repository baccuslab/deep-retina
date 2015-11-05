"""
Helper utilities

"""

from __future__ import print_function
from contextlib import contextmanager

__all__ = ['notify', 'rolling_window']


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

    print(title + '... ', end='', flush=True)
    try:
        yield
    finally:
        print('Done.')


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
