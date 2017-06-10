"""
fit an LN model
"""
import numpy as np
import os
from datetime import datetime
from keras.callbacks import (CSVLogger, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import BatchNormalization, Dense, Flatten, Input
from keras.metrics import cosine, mse
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from deepretina.activations import RBF, ParametricSoftplus, ReQU
from deepretina.experiments import loadexpt, _loadexpt_h5, cutout
from deepretina.metrics import kcc
from deepretina.models import bn_cnn

import pyret.filtertools as ft


def stimcut(data, width=9):
    # get the sta
    wn = _loadexpt_h5('15-10-07', 'whitenoise')
    sta = np.array(wn['train/stas/cell01']).copy()

    # peak of the sta
    xc, yc = ft.filterpeak(sta)[1]

    # slice indices
    xi = slice(xc - width, xc + width + 1)
    yi = slice(yc - width, yc + width + 1)

    return cutout(data, yi, xi)


def build(x, n_cells, l2pen=1.):
    y = Dense(n_cells, activation='softplus', kernel_regularizer=l2(l2pen))(Flatten()(x))
    # z = BatchNormalization(center=False, scale=False)(u)
    # y = RBF(30, 6)(z)
    # y = ReQU()(u)
    return Model(x, y, name='LN')


def train(expt, val_data, lr=1e-2):
    # name = 'BN_CNN_' + datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    name = 'LN_NS_' + datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    os.mkdir(f'../results/{name}')
    n_cells = expt.y.shape[1]

    x = Input(shape=(40, 50, 50), name='stimulus')
    model = build(x, n_cells)
    # x, y = bn_cnn((40, 50, 50), n_cells, l2_reg=0.05)
    # model = Model(x, y, name="BN_CNN")

    model.compile(loss='poisson', optimizer=Adam(lr), metrics=[kcc, cosine, mse])

    cbs = [ModelCheckpoint(os.path.join(f'../results/{name}', 'weights.{epoch:02d}-{val_loss:.2f}.h5')),
           TensorBoard(log_dir=f'../tensorboard/{name}'), ReduceLROnPlateau(min_lr=0, factor=0.2),
           CSVLogger(f'../results/{name}/training.csv')]

    return model.fit(x=expt.X, y=expt.y, batch_size=5000, epochs=500, callbacks=cbs, validation_data=val_data, shuffle=True)


if __name__ == '__main__':
    expt = '15-10-07'
    stim = 'naturalscene'

    data = loadexpt(expt, [0], stim, 'train', 40, 6000)
    cdata = stimcut(data)
    val_data = loadexpt(expt, [0], stim, 'test', 40, 0)
    cval_data = stimcut(val_data)

    history = train(data, val_data)
