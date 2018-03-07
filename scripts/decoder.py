# %load_ext autoreload
# %autoreload 1

import numpy as np
import collections
import pyret.filtertools as ft
import tensorflow as tf
load_model = tf.keras.models.load_model
from pathlib import Path
import h5py
import skvideo.io
import os
import functools

K = tf.keras.backend
from deepretina import config

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
# config.results_dir = "/home/tyler/results/"
config.results_dir = "/home/tyler/results/"
config.data_dir = "/home/salamander/experiments/data/"

weights = Path(config.results_dir) / "G-CNN__15-11-21b_naturalscene_2018.02.18-22.28.34" / "weights-171--6.156.h5"
model = load_model(weights, custom_objects={"tf": tf, "ResizeMethod": D.models.ResizeMethod,
    "argmin_poisson": D.metrics.argmin_poisson,
    "matched_cc": D.metrics.matched_cc,
    "matched_mse": D.metrics.matched_mse,
    "matched_rmse": D.metrics.matched_rmse,
    })


datafile = Path(config.data_dir) / "15-11-21b" / "naturalscene.h5"
data = h5py.File(datafile, mode='r')

# %%

x = np.array(data["train"]["stimulus"])
y_train = x[41:]
x = D.experiments.rolling_window(x,40)
x = x[0:len(x)-1]
n = len(x)
x_train = model.predict(x, batch_size=1000)
x_train.shape
ntrain = int(np.floor(n*0.95))
nvalid = n - ntrain
train_data = ExptSequence([
    D.experiments.Exptdata(x_train[0:ntrain],y_train[0:ntrain])
    ])
valid_data = ExptSequence([
    D.experiments.Exptdata(x_train[ntrain:ntrain+nvalid],y_train[ntrain:ntrain+nvalid])
    ])
run_name = "Decoder_naturalmovie"
input_shape = train_data[0][0].shape[1:]
train_data[0][1].shape
steps_per_epoch = len(train_data)
steps_per_valid = len(valid_data)

# %%
@D.utils.context
def fit_decoder(train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid):
    D.core.train_generator(D.models.decoder, train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[], loss='mse')

# %%
fit_decoder(train_data, input_shape, steps_per_epoch, run_name, valid_data, steps_per_valid)


# @D.utils.context
# def fit_autoencoder(train_data, valid_data, name="AE"):
#     D.core.simple_train(D.models.auto_encoder, train_data, valid_data, name, lr=1e-2, nb_epochs=250, bz=1000,the_metrics=[])
#
# # %%
# fit_autoencoder(train_data, valid_data, "AE_test")
