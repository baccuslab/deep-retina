"""
Core tools for training models
"""
import os
from datetime import datetime
import deepdish as dd
import tensorflow as tf
keras = tf.keras
cb = keras.callbacks
Input = tf.keras.layers.Input
from deepretina import metrics, activations
from deepretina.experiments import loadexpt, CELLS, Exptdata
load_model = tf.keras.models.load_model
Adam = tf.keras.optimizers.Adam
from deepretina import config
import numpy as np


__all__ = ['train', 'load']


def load(filepath):
    """Reload a keras model"""
    objects = {k: activations.__dict__[k] for k in activations.__all__}
    objects.update({k: metrics.__dict__[k] for k in metrics.__all__})
    return load_model(filepath, custom_objects=objects)


def train(model, expt, stim, model_args=(), lr=1e-2, bz=5000, nb_epochs=500, val_split=0.05, cells=None, loss="poisson", train_data=None,valid_data=None,the_metrics=[metrics.cc, metrics.rmse, metrics.fev]):
    """Trains a model"""
    if cells is None:
        width = None
        cells = CELLS[expt]
        cellname = ''
    else:
        width = 11
        cellname = f'cell-{cells[0]+1:02d}'

    # load experimental data
    if train_data==None:
        data = loadexpt(expt, cells, stim, 'train', 40, 6000, cutout_width=width)
        n = data.X.shape[0]
        ntrain = int(np.floor(n*0.95))
        train_data = Exptdata(data.X[0:ntrain],data.y[0:ntrain])
        valid_data = Exptdata(data.X[ntrain:],data.y[ntrain:])

    # build the model
    n_cells = train_data.y.shape[1]
    x = Input(shape=train_data.X.shape[1:])
    mdl = model(x, n_cells, *model_args)

    # compile the model
    mdl.compile(loss=loss, optimizer=Adam(lr), metrics=the_metrics)

    # store results in this directory
    name = '_'.join([mdl.name, cellname, expt, stim, datetime.now().strftime('%Y.%m.%d-%H.%M.%S')])
    base = config.results_dir + name
    os.mkdir(base)

    # define model callbacks
    cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
           # cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=20)]

    # train
    history = mdl.fit(x=train_data.X, y=train_data.y, batch_size=bz, epochs=nb_epochs,
        callbacks=cbs, validation_data=valid_data, shuffle=True)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)

    return history

def simple_train(model, train_data, valid_data, run_name=None, model_args=(), lr=1e-2, bz=1000, nb_epochs=250, loss="poisson", the_metrics=[metrics.cc, metrics.rmse, metrics.fev]):
    """Trains a model"""

    # build the model
    x = Input(shape=train_data.X.shape[1:])
    mdl = model(x, *model_args)

    # compile the model
    mdl.compile(loss=loss, optimizer=Adam(lr), metrics=the_metrics)

    # store results in this directory
    name = '_'.join([mdl.name, run_name, datetime.now().strftime('%Y.%m.%d-%H.%M.%S')])
    base = config.results_dir + name
    os.mkdir(base)

    # define model callbacks
    cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
           cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=20)]

    # train
    history = mdl.fit(x=train_data.X, y=train_data.y, batch_size=bz, epochs=nb_epochs,
        callbacks=cbs, validation_data=valid_data, shuffle=True)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)

    return history


def train_generator(model, train_gen, input_shape, steps_per_epoch, run_name, valid_gen=None, validation_steps=None, model_args=(), lr=1e-2, bz=1000, nb_epochs=250, loss="poisson", the_metrics=[metrics.cc, metrics.rmse, metrics.fev]):
    """Trains a model"""
    # build the model
    x = Input(shape=input_shape)
    mdl = model(x, *model_args)

    # compile the model
    mdl.compile(loss=loss, optimizer=Adam(lr), metrics=the_metrics)

    # store results in this directory
    name = '_'.join([mdl.name, run_name,
        datetime.now().strftime('%Y.%m.%d-%H.%M.%S')])
    base = config.results_dir + name
    os.mkdir(base)

    # define model callbacks
    cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
           # cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True),
           cb.TensorBoard(log_dir=base, batch_size=1000, write_grads=True),
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=20)]

    # train
    history = mdl.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
        epochs=nb_epochs, callbacks=cbs, validation_data=valid_gen,
        validation_steps=validation_steps, shuffle=False)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)

    return history
