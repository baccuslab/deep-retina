import numpy as np
from collections import namedtuple


Score = namedtuple('Score', ['cc', 'lli', 'rmse', 'fev'])
scorenames = {
    'cc': 'Correlation Coefficient',
    'lli': 'Log-likelihood improvement (bits / spike)',
    'rmse': 'Root mean squared error',
    'fev': 'Frac. of explained variance'
}


def cc(r, rhat):
    """
    Correlation coefficient
    """
    return np.corrcoef(np.vstack((rhat, r)))[0, 1]


def lli(r, rhat):
    """
    Log-likelihood improvement over a mean rate model (in bits per spike)
    """

    mean = np.mean(rhat)
    mu = float(np.mean(r * np.log(mean) - mean))
    return (np.mean(r * np.log(rhat) - rhat) - mu) / (mean * np.log(2))


def rmse(r, rhat):
    """
    Root mean squared error
    """
    return np.sqrt(np.mean((rhat - r) ** 2))


def fev(r, rhat):
    """
    Fraction of explained variance
    """

    mean = np.mean(r)
    rate_var = np.mean((mean - r) ** 2)
    return 1.0 - (rmse(r, rhat) ** 2) / rate_var
