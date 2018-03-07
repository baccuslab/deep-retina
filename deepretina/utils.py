"""
Generic utilities
"""
import sys
import tensorflow as tf
K = tf.keras.backend
from contextlib import contextmanager

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


def cutout_indices(center, size=7, ndim=50):
    """Cuts out a region with the given size around a point"""
    xinds = slice(int(np.clip(center[0] - size, 0, ndim)),
                  int(np.clip(center[0] + size + 1, 0, ndim)))
    yinds = slice(int(np.clip(center[1] - size, 0, ndim)),
                  int(np.clip(center[1] + size + 1, 0, ndim)))
    return xinds, yinds

def context(func):
    def wrapper(*args, **kwargs):
        with tf.device('/gpu:0'):
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session(graph=graph) as sess:
                    K.set_session(sess)
                    result = func(*args, **kwargs)
            return result
    return wrapper
