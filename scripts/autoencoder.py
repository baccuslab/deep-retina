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
from pathlib import Path
import h5py
#
import deepretina
import deepretina.experiments
import deepretina.models
import deepretina.core
import deepretina.metrics
import deepretina.utils

# %aimport deepretina
# %aimport deepretina.experiments
# %aimport deepretina.models
# %aimport deepretina.core
# %aimport deepretina.metrics
# %aimport deepretina.utils
D = deepretina
ExptSequence = D.experiments.ExptSequence
# config.results_dir = "/home/tyler/scratch/results/"
config.results_dir = "/home/tyler/results/"


config.data_dir = "/home/salamander/experiments/data/"
datafile = Path(config.data_dir) / "17-12-09b-ssb" / "naturalmovie.h5"
data = h5py.File(datafile, mode='r')

# %%

X = np.array(data["train"]["stimulus"])
Y = X[41:]
X = D.experiments.rolling_window(X,40)
X = X[0:len(X)-1]
n = len(X)
ntrain = int(np.floor(n*0.9))
nvalid = int(np.floor(n*0.05))
ntest = n - ntrain - nvalid
train_data = ExptSequence([
    D.experiments.Exptdata(X[0:ntrain],Y[0:ntrain])
    ])
valid_data = ExptSequence([
    D.experiments.Exptdata(X[ntrain:ntrain+nvalid],Y[ntrain:ntrain+nvalid])
    ])
run_name = "auto_encoder_naturalmovie"
input_shape = train_data[0][0].shape[1:]
train_data[0][1].shape
steps_per_epoch = len(train_data)
steps_per_valid = len(valid_data)

# %%
@D.utils.context
def fit_autoencoder(train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid):
    D.core.train_generator(D.models.auto_encoder, train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[], loss='mse')

# %%
fit_autoencoder(train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid)


# @D.utils.context
# def fit_autoencoder(train_data, valid_data, name="AE"):
#     D.core.simple_train(D.models.auto_encoder, train_data, valid_data, name, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[])
#
# # %%
# fit_autoencoder(train_data, valid_data, "AE_test")
