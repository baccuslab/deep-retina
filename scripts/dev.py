%load_ext autoreload
%autoreload 1

import os
import functools
import argparse
import tensorflow as tf
import keras.backend as K
import tableprint as tp
import numpy as np
from deepretina import config
%aimport deepretina
%aimport deepretina.experiments
%aimport deepretina.models
%aimport deepretina.core
%aimport deepretina.metrics
D = deepretina

expt = "15-11-21b"
stim = "whitenoise"
cells = D.experiments.CELLS[expt]

config.data_dir = "/storage/baccus/"
config.results_dir = "/storage/baccus/results"
data = D.experiments.loadexpt(expt, cells, stim, 'train', 40, 6000)
data.X.shape
y = tf.reshape(data.y,[*data.y.shape,1,1])
newdata = D.experiments.Exptdata(data.X,y)
D.core.train(D.models.g_cnn, expt, stim, lr=1e-2, nb_epochs=1, val_split=0.05,data=newdata,loss=D.metrics.argmin_loss)
# D.core.train(D.models.bn_cnn, expt, stim, lr=1e-2, nb_epochs=1, val_split=0.05,data=data,loss=D.metrics.poisson)

%%

fit_g_cnn(expt, stim)

np.min(np.arange(12).reshape(4,3))
