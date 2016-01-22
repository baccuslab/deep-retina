"""
Metrics comparing predicted and recorded firing rates

"""

import numpy as np
from scipy.stats import pearsonr

__all__ = ['cc', 'lli', 'rmse', 'fev']


def cc(r, rhat):
    """
    Correlation coefficient
    """

    return pearsonr(r, rhat)[0]


def lli(r, rhat):
    """
    Log-likelihood improvement over a mean rate model (in bits per spike)
    """

    mu = np.mean(rhat)
    mu = float(np.mean(r * np.log(mu) - mu))
    return (np.mean(r * np.log(rhat) - rhat) - mu) / (mu * np.log(2))


def rmse(r, rhat):
    """
    Root mean squared error
    """

    return np.sqrt(np.mean((rhat - r) ** 2))


def fev(r, rhat):
    """
    Fraction of explained variance

    wikipedia.org/en/Fraction_of_variance_unexplained
    """

    return 1.0 - rmse(r, rhat) / r.var()
