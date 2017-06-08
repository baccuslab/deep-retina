"""
Metrics comparing predicted and recorded firing rates
"""

from __future__ import absolute_import, division, print_function

from functools import wraps

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import auc
from tqdm import tqdm
import keras.backend as K

__all__ = ['cc', 'lli', 'rmse', 'fev']


def kcc(y_true, y_pred):
    x_mu = y_true - K.mean(y_true, axis=0, keepdims=True)
    x_std = K.std(y_true, axis=0, keepdims=True)
    y_mu = y_pred - K.mean(y_pred, axis=0, keepdims=True)
    y_std = K.std(y_pred, axis=0, keepdims=True)
    return K.mean(x_mu * y_mu, axis=0, keepdims=True) / (x_std * y_std)


def multicell(metric):
    """Decorator for turning a function that takes two 1-D numpy arrays, and
    makes it work for when you have a list of 1-D arrays or a 2-D array, where
    the metric is applied to each item in the list or each matrix row.
    """
    @wraps(metric)
    def multicell_wrapper(obs_rate, est_rate, **kwargs):

        # ensure that the arguments have the right shape / dimensions
        for arg in (obs_rate, est_rate):
            assert isinstance(arg, (np.ndarray, list, tuple)), \
                "Arguments must be a numpy array or list of numpy arrays"

        # convert arguments to matrices
        true_rates = np.atleast_2d(obs_rate)
        model_rates = np.atleast_2d(est_rate)

        assert true_rates.ndim == 2, "Arguments have too many dimensions"
        assert true_rates.shape == model_rates.shape, "Shapes must be equal"

        # compute scores for each pair
        scores = [metric(true_rate, model_rate)
                  for true_rate, model_rate in zip(true_rates, model_rates)]

        # return the mean across cells and the full list
        return np.nanmean(scores), scores

    return multicell_wrapper


@multicell
def cc(obs_rate, est_rate):
    """Pearson's correlation coefficient

    If obs_rate, est_rate are matrices, cc() computes the average pearsonr correlation
    of each column vector
    """
    return pearsonr(obs_rate, est_rate)[0]


@multicell
def lli(obs_rate, est_rate):
    """Log-likelihood (arbitrary units)"""
    epsilon = 1e-9
    return np.mean(obs_rate * np.log(est_rate + epsilon) - est_rate)


@multicell
def rmse(obs_rate, est_rate):
    """Root mean squared error"""
    return np.sqrt(np.mean((est_rate - obs_rate) ** 2))


@multicell
def fev(obs_rate, est_rate):
    """Fraction of explained variance

    https://wikipedia.org/en/Fraction_of_variance_unexplained
    """
    return 1.0 - rmse(obs_rate, est_rate)[0]**2 / obs_rate.var()


def roc(obs_rate, est_rate):
    """Generates an ROC curve"""
    thresholds = np.linspace(0, 100, 1e2)
    data = np.vstack([binarized(obs_rate, est_rate, thr) for thr in tqdm(thresholds)])
    fpr = data[:, 0]
    tpr = data[:, 1]
    tpr[np.isnan(tpr)] = 0.     # nans should be zero
    return fpr, tpr, auc(fpr, tpr, reorder=True)


def binarized(obs_rate, est_rate, threshold):
    """Computes fraction of correct predictions given the threshold"""
    bin_obs_rate = obs_rate > threshold
    bin_est_rate = est_rate > threshold

    true_positive = sum(bin_obs_rate & bin_est_rate)
    true_negative = sum(np.invert(bin_obs_rate) & np.invert(bin_est_rate))
    false_positive = sum(np.invert(bin_obs_rate) & bin_est_rate)
    false_negative = sum(bin_obs_rate & np.invert(bin_est_rate))

    true_positive_rate = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)

    return false_positive_rate, true_positive_rate
