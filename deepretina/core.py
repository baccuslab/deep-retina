"""
Core tools for training models
"""
import os
from datetime import datetime

import deepdish as dd
import keras.callbacks as cb
from keras.layers import Input
from deepretina import metrics, activations
from deepretina.experiments import loadexpt, CELLS
from keras.models import load_model
from keras.optimizers import Adam

__all__ = ['train', 'load']


def load(filepath):
    """Reload a keras model"""
    objects = {k: activations.__dict__[k] for k in activations.__all__}
    objects.update({k: metrics.__dict__[k] for k in metrics.__all__})
    return load_model(filepath, custom_objects=objects)


def train(model, expt, stim, lr=1e-2, bz=5000, nb_epochs=500, val_split=0.05):
    """Trains a model"""

    # build the model
    data = loadexpt(expt, CELLS[expt], stim, 'train', 40, 6000)
    n_cells = data.y.shape[1]
    x = Input(shape=data.X.shape[1:], name='stimulus')
    mdl = model(x, n_cells)

    # compile the model
    mdl.compile(loss='poisson', optimizer=Adam(lr), metrics=[metrics.cc, metrics.rmse, metrics.fev])

    # store results in this directory
    name = '_'.join([mdl.name, expt, stim, datetime.now().strftime('%Y.%m.%d')])
    base = f'../results/{name}'
    os.mkdir(base)

    # define model callbacks
    cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights.{epoch:02d}-{val_loss:.2f}.h5')),
           cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=10)]

    # train
    history = mdl.fit(x=data.X, y=data.y, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_split=val_split, shuffle=True)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)
    return history
