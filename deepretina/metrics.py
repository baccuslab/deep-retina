"""
Metrics comparing predicted and recorded firing rates

All metrics in this module are computed separately for each cell, but averaged
across the sample dimension (axis=0).
"""
from functools import wraps

import tensorflow as tf
import keras.backend as K
from keras import layers

__all__ = ['cc', 'rmse', 'fev', 'CC', 'RMSE', 'FEV', 'np_wrap',
           'root_mean_squared_error', 'correlation_coefficient',
           'fraction_of_explained_variance']


def correlation_coefficient(obs_rate, est_rate):
    """Pearson correlation coefficient"""
    x_mu = obs_rate - K.mean(obs_rate, axis=0, keepdims=True)
    x_std = K.std(obs_rate, axis=0, keepdims=True)
    y_mu = est_rate - K.mean(est_rate, axis=0, keepdims=True)
    y_std = K.std(est_rate, axis=0, keepdims=True)
    return K.mean(x_mu * y_mu, axis=0, keepdims=True) / (x_std * y_std)


def mean_squared_error(obs_rate, est_rate):
    """Mean squared error across samples"""
    return K.mean(K.square(est_rate - obs_rate), axis=0, keepdims=True)


def root_mean_squared_error(obs_rate, est_rate):
    """Root mean squared error"""
    return K.sqrt(mean_squared_error(obs_rate, est_rate))


def fraction_of_explained_variance(obs_rate, est_rate):
    """Fraction of explained variance

    https://wikipedia.org/en/Fraction_of_variance_unexplained
    """
    return 1.0 - mean_squared_error(obs_rate, est_rate) / K.var(obs_rate, axis=0, keepdims=True)

def poisson(y_true, y_pred):
    loss = K.mean(tf.subtract(y_pred, y_true) * K.log(tf.add(y_pred, K.epsilon())), axis=-1)
    # print(tf.shape(loss),tf.shape(y_true),tf.shape(y_pred))
    return loss

def argmin_loss(obs_rate, est_rate):
    """Find a matching that produces the lowest loss and return that loss.
    Params:
        obs_rate: B x R
        est_rate: B x X x Y x C
    Return:
        loss: L
    future ref: https://github.com/mbaradad/munkres-tensorflow"""
    # print("est_rate",est_rate.shape)
    # print("obs_rate",obs_rate.shape)
    nr = tf.shape(obs_rate)[1]
    nb = tf.shape(est_rate)[0]
    nx = tf.shape(est_rate)[1]
    ny = tf.shape(est_rate)[2]
    nc = tf.shape(est_rate)[3]
    obs = tf.reshape(obs_rate, [nb,nr,1])
    est = tf.reshape(est_rate, (nb,1,nx*ny*nc))
    # print("est",est.shape)
    # print("obs",obs.shape)

    # print("!!!",tf.shape(tf.subtract(est, obs)))
    # B x R x (X+Y+C)
    loss = tf.multiply(
        tf.subtract(est, obs),
        K.log(tf.add(est, K.epsilon())))
    # R x (X+Y+C)
    mean_loss = tf.reduce_mean(loss,axis=[0])
    # R
    argmin = tf.reduce_min(mean_loss,axis=[1])
    # print("argmin",tf.shape(argmin))
    # B x R x (X+Y+C)
    # tf.segment_min(obs_rate,tf.ones(obs_rate.shape))
    # min(N x N)
    # matching = 1
    return K.mean(argmin)


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
