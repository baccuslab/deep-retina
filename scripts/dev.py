# %load_ext autoreload
# %autoreload 1

import os
import functools
import argparse
import tensorflow as tf
K = tf.keras.backend
import tableprint as tp
import numpy as np
from deepretina import config
import deepretina
import deepretina.experiments
import deepretina.models
import deepretina.core
import deepretina.metrics
# %aimport deepretina
# %aimport deepretina.experiments
# %aimport deepretina.models
# %aimport deepretina.core
# %aimport deepretina.metrics
D = deepretina

expt = "15-11-21b"
stim = "whitenoise"
# stim = "naturalscene"
cells = D.experiments.CELLS[expt]

config.data_dir = "/home/salamander/experiments/data/"
config.results_dir = "/home/tyler/results/"
# data = D.experiments.loadexpt(expt, cells, stim, 'train', 40, 6000)
# n = data.X.shape[0]
# ntrain = int(np.floor(n*0.95))
# y = np.reshape(data.y,[*data.y.shape,1,1])
# data.X[0:10].shape
# train_data = D.experiments.Exptdata(data.X[0:ntrain],y[0:ntrain])
# valid_data = D.experiments.Exptdata(data.X[ntrain:],y[ntrain:])
#
# small_train_data = D.experiments.Exptdata(train_data.X[:2000],train_data.y[:2000])
# small_valid_data = D.experiments.Exptdata(valid_data.X[:1000],valid_data.y[:1000])


def context(func):
    def wrapper(*args, **kwargs):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                K.set_session(sess)
                result = func(*args, **kwargs)
        return result
    return wrapper

# %%
@context
def fit_g_cnn(expt, stim, train_data, valid_data):
    D.core.train(D.models.g_cnn, expt, stim, lr=1e-2, nb_epochs=250, bz=1000, train_data=train_data, valid_data=valid_data,    loss=D.metrics.argmin_poisson,
    the_metrics=[D.metrics.matched_cc, D.metrics.matched_mse, D.metrics.matched_rmse])


@context
def fit_bn_cnn(expt, stim):
    D.core.train(D.models.bn_cnn, expt, stim, lr=1e-2, nb_epochs=250, val_split=0.05)

# fit_g_cnn(expt, stim, train_data, small_valid_data)

fit_bn_cnn(expt, stim)
