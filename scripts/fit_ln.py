import os
import datetime
from keras.callbacks import (CSVLogger, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import BatchNormalization, Dense, Flatten, Input
from keras.metrics import cosine, mse
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from deepretina.activations import RBF, ParametricSoftplus, ReQU
from deepretina.experiments import loadexpt
from deepretina.metrics import kcc


def build(x, n_cells, l2pen=1.):
    u = Dense(n_cells, kernel_regularizer=l2(l2pen))(Flatten()(x))
    # z = BatchNormalization(center=False, scale=False)(u)
    # y = RBF(30, 6)(z)
    y = ReQU(u)
    return Model(x, y, name='LN')


def train(expt):
    name = 'LN_' + datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    os.mkdir(f'../results/{name}')
    n_cells = expt.y.shape[1]
    x = Input(shape=(40, 50, 50), name='stimulus')
    model = build(x, n_cells)
    model.compile(loss='poisson', optimizer=Adam(1e-2), metrics=[kcc, cosine, mse])
    cbs = [ModelCheckpoint('../results/{name}/weights.{epoch:02d}-{val_loss:.2f}.h5'), TensorBoard(log_dir=f'../tensorboard/{name}'), ReduceLROnPlateau(min_lr=1e-5), CSVLogger(f'../results/{name}/training.csv')]
    return model.fit(x=expt.X, y=expt.y, batch_size=5000, epochs=50, callbacks=cbs, validation_split=0.05, shuffle=True)


if __name__ == '__main__':
    expt = loadexpt('15-10-07', [0, 1, 2, 3, 4], 'whitenoise', 'train', 40, 6000)
    history = train(expt)
