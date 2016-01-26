"""
Metrics comparing predicted and recorded firing rates

"""

import numpy as np
from scipy.stats import pearsonr

__all__ = ['cc', 'lli', 'rmse', 'fev', 'multicell']


def multicell(function):
    """
    This function returns a new function which can be used to apply
    a metric function to a bunch of cells, returning the score for each pair
    as well as the average

    """

    def metric_multicell(r, rhat):
        assert r.shape == rhat.shape, "Argument shapes must be equal"
        assert r.ndim == 2, "Inputs must be matrices"

        scores = [function(r[:, cell_index], rhat[:, cell_index])
                  for cell_index in range(r.shape[1])]

        return np.mean(scores), scores

    return metric_multicell


def cc(r, rhat):
    """
    Correlation coefficient. By default averages over

    If r, rhat are matrices, cc() computes the average pearsonr correlation
    of each column vector
    """
    assert r.shape == rhat.shape, "Argument shapes must be equal"
    assert r.ndim == 1, "Inputs must be vectors"

    return pearsonr(r, rhat)[0]


def lli(r, rhat):
    """
    Log-likelihood improvement over a mean rate model (in bits per spike)
    """
    assert r.shape == rhat.shape, "Argument shapes must be equal"
    assert r.ndim == 1, "Inputs must be vectors"

    mu = np.mean(rhat)
    mu = float(np.mean(r * np.log(mu) - mu))
    return (np.mean(r * np.log(rhat) - rhat) - mu) / (mu * np.log(2))


def rmse(r, rhat):
    """
    Root mean squared error
    """
    assert r.shape == rhat.shape, "Argument shapes must be equal"
    assert r.ndim == 1, "Inputs must be vectors"

    return np.sqrt(np.mean((rhat - r) ** 2))


def fev(r, rhat):
    """
    Fraction of explained variance

    wikipedia.org/en/Fraction_of_variance_unexplained
    """
    assert r.shape == rhat.shape, "Argument shapes must be equal"
    assert r.ndim == 1, "Inputs must be vectors"

    return 1.0 - rmse(r, rhat) / r.var()
