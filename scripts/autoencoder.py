%load_ext autoreload
%autoreload 1

import os
import functools
import argparse
import tensorflow as tf
K = tf.keras.backend
import tableprint as tp
import numpy as np
from deepretina import config
from pathlib import Path
import h5py
#
# import deepretina
# import deepretina.experiments
# import deepretina.models
# import deepretina.core
# import deepretina.metrics
# import deepretina.utils

%aimport deepretina
%aimport deepretina.experiments
%aimport deepretina.models
%aimport deepretina.core
%aimport deepretina.metrics
%aimport deepretina.utils
D = deepretina
config.results_dir = "/home/tyler/scratch/results/"

datafile = Path("/storage/baccus/17-12-09b-ssb") / "naturalmovie.h5"
data = h5py.File(datafile, mode='r')
data.attrs

# %%
def window_ndarray(array,window_width,step=1):
        n = len(array)
        nsteps = int(np.floor((n-window_width)/step)) + 1
        idx = np.arange(window_width)[None, :] + step*np.arange(nsteps)[:, None]
        return array[idx]
X = data["train"]["stimulus"]
Y = X[41:]
# X = window_ndarray(X,40)
X = D.experiments.rolling_window(X,40)
X = X[0:len(X)-1]
X.shape
Y.shape

n = len(X)
ntrain = int(np.floor(n*0.95))
nvalid = n - ntrain
train_data = D.experiments.Exptdata(X[0:ntrain],Y[0:ntrain])
valid_data = D.experiments.Exptdata(X[ntrain:ntrain+1000],Y[ntrain:ntrain+1000])
run_name = "auto_encoder_naturalmovie"
# %%
@D.utils.context
def fit_autoencoder(train_data, valid_data, name="AE"):
    D.core.simple_train(D.models.auto_encoder, train_data, valid_data, name, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[])

# %%
fit_autoencoder(train_data, valid_data, "AE_test")


# @D.utils.context
# def fit_autoencoder(train_data, valid_data, name="AE"):
#     D.core.simple_train(D.models.auto_encoder, train_data, valid_data, name, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[])
#
# # %%
# fit_autoencoder(train_data, valid_data, "AE_test")
