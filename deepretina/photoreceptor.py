"""
Simple model for photoreceptor adaptation

Based on the paper:
Dynamical Adaptation in Photoreceptors (Clark et. al. 2015)
http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.special import gamma


def pr_filter(dt, stim, tau_y=0.033, ny=4., tau_z=0.019, nz=10., alpha=1., beta=0.16, eta=0.23):
    """Filter the given stimulus using a model of photoreceptor adaptation

    """

    # build the two filters
    t = np.arange(dt, 0.5, dt)
    Ky = dt * _make_filter(t, tau_y, ny)
    Kz = eta * Ky + (1 - eta) * dt * _make_filter(t, tau_z, nz)

    # filter the stimulus
    y = np.zeros_like(stim)
    z = np.zeros_like(stim)
    T = stim.shape[0]
    for row in range(stim.shape[1]):
        for col in range(stim.shape[2]):
            y[:, row, col] = np.convolve(stim[:, row, col], Ky, mode='full')[:T]
            z[:, row, col] = np.convolve(stim[:, row, col], Kz, mode='full')[:T]

    # return the filtered stimulus
    return (alpha * y) / (1 + beta * z)


def _make_filter(t, tau, n):
    """Makes a simple filter with the given time constant and exponent

    Parameters
    ----------
    t : array_like
        time array

    tau : float
        time constant

    n : int
        exponent

    """

    return ((t ** n) / (gamma(n + 1) * tau ** (n + 1))) * np.exp(-t / tau)
