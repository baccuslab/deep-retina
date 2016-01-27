"""
Metrics comparing predicted and recorded firing rates

"""

import numpy as np
from scipy.stats import pearsonr

__all__ = ['cc', 'lli', 'rmse', 'fev', 'multicell']


def multicell(metric):
    """
    This function returns a new function which when called will apply the
    given metric across the columns of the input array, returning the score
    for each pair as well as the average

    """

    def metric_multicell(r, rhat):
        assert r.shape == rhat.shape, "Argument shapes must be equal"
        assert r.ndim == 2, "Inputs must be matrices"

        scores = [metric(r[:, cell_index], rhat[:, cell_index])
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


def lli(r, rhat, dt=1e-2):
    """
    Log-likelihood improvement over a mean rate model (in bits per spike)

    Parameters
    ----------

    r : array_like
        True firing rate

    rhat : array_like
        Model firing rate

    dt : float, optional
        Sampling period (in seconds). Default: 0.010s

    """
    assert r.shape == rhat.shape, "Argument shapes must be equal"
    assert r.ndim == 1, "Inputs must be vectors"

    # mean firing rate
    mu = np.mean(r)

    # poisson log-likelihood
    def loglikelihood(q):
        return r * np.log(q) - q

    # difference in log-likelihoods (in bits per spike)
    return np.mean(loglikelihood(rhat) - loglikelihood(mu)) / (mu * np.log(2))


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
