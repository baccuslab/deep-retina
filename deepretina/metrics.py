"""
Metrics comparing predicted and recorded firing rates

All metrics in this module are computed separately for each cell, but averaged
across the sample dimension (axis=0).
"""
from functools import wraps

import tensorflow as tf
K = tf.keras.backend
from tensorflow.python.keras import layers
import numpy as np

__all__ = ['cc', 'rmse', 'fev', 'CC', 'RMSE', 'FEV', 'np_wrap',
           'root_mean_squared_error', 'correlation_coefficient',
           'fraction_of_explained_variance']


def correlation_coefficient(obs_rate, est_rate):
    """Pearson correlation coefficient"""
    obs_mu = obs_rate - K.mean(obs_rate, axis=0, keepdims=True)
    obs_std = K.std(obs_rate, axis=0, keepdims=True)
    est_mu = est_rate - K.mean(est_rate, axis=0, keepdims=True)
    est_std = K.std(est_rate, axis=0, keepdims=True)
    return K.mean(obs_mu * est_mu, axis=0, keepdims=True) / (obs_std * est_std + K.epsilon())

def mean_squared_error(obs_rate, est_rate):
    """Mean squared error across samples"""
    return K.mean(K.square(est_rate - obs_rate), axis=0, keepdims=True)


def root_mean_squared_error(obs_rate, est_rate):
    """Root mean squared error"""
    return K.sqrt(mean_squared_error(obs_rate, est_rate))

def get_gconv_shape(obs_rate, est_rate):
    "Return dims and reshape to obs: B x R x 1, est: B x 1 x (X*Y*C)"
    nr = tf.shape(obs_rate)[1]
    nb = tf.shape(est_rate)[0]
    nc = tf.shape(est_rate)[1]
    nx = tf.shape(est_rate)[2]
    ny = tf.shape(est_rate)[3]
    obs = tf.reshape(obs_rate, [nb,nr,1])
    est = tf.reshape(est_rate, (nb,1,nx*ny*nc))
    return obs, est, nb, nr, nx, ny, nc


def matched_cc(obs_rate, est_rate):
    """Pearson correlation coefficient"""

    obs_matched, est_matched = poisson_matching(obs_rate,est_rate)
    return correlation_coefficient(obs_matched,est_matched)

def matched_mse(obs_rate, est_rate):
    """Mean squared error across samples"""
    obs_matched, est_matched = poisson_matching(obs_rate,est_rate)
    return mean_squared_error(obs_matched,est_matched)


def matched_rmse(obs_rate, est_rate):
    """Root mean squared error"""
    obs_matched, est_matched = poisson_matching(obs_rate,est_rate)
    return root_mean_squared_error(obs_matched,est_matched)

def fraction_of_explained_variance(obs_rate, est_rate):
    """Fraction of explained variance

    https://wikipedia.org/en/Fraction_of_variance_unexplained
    """
    return 1.0 - mean_squared_error(obs_rate, est_rate) / K.var(obs_rate, axis=0, keepdims=True)

def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)

def poisson_matching(y_true, y_pred):
    """Find a matching that produces the lowest loss and return that matching.
    Params:
        y_true: B x R
        y_pred: B x X x Y x C
    Return:
        matching: (B x R, B x R)
    """
    obs, est, nb, nr, nx, ny, nc = get_gconv_shape(y_true,y_pred)
    loss = est - obs * K.log(est + K.epsilon())
    # R x (X+Y+C)
    mean_loss = tf.reduce_mean(loss,axis=[0])
    # reduce across XYC
    matching = tf.argmin(mean_loss,axis=1)
    est_matched = tf.gather(est[:,0,:],matching,axis=1)
    obs_matched = obs[:,:,0]
    return obs_matched, est_matched

def poisson_argmin_idx(y_true, y_pred):
    "Return optimal (c, x, y) mactching per y_true using CPU (numpy)."
    nr = np.shape(y_true)[1]
    nb = np.shape(y_pred)[0]
    nc = np.shape(y_pred)[1]
    nx = np.shape(y_pred)[2]
    ny = np.shape(y_pred)[3]
    obs = np.reshape(y_true, [nb,nr,1])
    est = np.reshape(y_pred, (nb,1,nx*ny*nc))

    loss = est - obs * np.log(est + 1e-07)
    # R x (X+Y+C)
    mean_loss = np.mean(loss,0)
    # reduce across XYC
    matching = np.argmin(mean_loss,axis=1)
    matching = np.array(np.unravel_index(matching,np.shape(y_pred)[1:])).T
    return matching

def argmin_poisson(y_true, y_pred):
    """Find a matching that produces the lowest loss and return that loss.
    Params:
        y_true: B x R
        y_pred: B x X x Y x C
    Return:
        loss: L
    """
    obs, est, nb, nr, nx, ny, nc = get_gconv_shape(y_true,y_pred)

    loss = est - obs * K.log(est + K.epsilon())
    # R x (X+Y+C)
    mean_loss = tf.reduce_mean(loss,axis=[0])
    # R
    minimized = tf.reduce_min(mean_loss,axis=[1])
    return K.mean(minimized)


def np_wrap(func):
    """Converts the given keras metric into one that accepts numpy arrays

    Usage
    -----
    corrcoef = np_wrap(cc)(y, yhat)     # y and yhat are numpy arrays
    """
    @wraps(func)
    def wrapper(obs_rate, est_rate):
        with tf.Session() as sess:
            # compute the metric
            yobs = tf.placeholder(tf.float64, obs_rate.shape)
            yest = tf.placeholder(tf.float64, est_rate.shape)
            metric = func(yobs, yest)

            # evaluate using the given values
            feed = {yobs: obs_rate, yest: est_rate}
            return sess.run(metric, feed_dict=feed)
    return wrapper


# aliases
cc = CC = correlation_coefficient
rmse = RMSE = root_mean_squared_error
fev = FEV = fraction_of_explained_variance
