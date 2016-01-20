"""
Toy demo for sequence prediction using LSTM layers
Uses Keras v0.3

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import adam, SGD
from datagen import generate_batch
import tableprint as tp
import numpy as np


def build(n_input, n_hidden):

    model = Sequential()

    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(n_input, 1)))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer=adam())

    return model


def train(niter, nsteps, nhidden, ntrain, ntest, snr=10.0):

    # build a model
    model = build(nsteps, nhidden)

    # generate a test batch
    Xtest, ytest = generate_batch(ntest, nsteps, snr=snr)
    Xtest = Xtest.reshape(*Xtest.shape, 1)

    # print header
    print(tp.header(['Iter', 'Train', 'Test']))

    error = []

    for idx in range(niter):

        # get data
        X, y = generate_batch(ntrain, nsteps, snr=snr)
        X = X.reshape(*X.shape, 1)

        # train
        fobj = model.train_on_batch(X, y)

        # test
        yhat = model.predict(Xtest).ravel()
        test_err = ((yhat - ytest) ** 2).mean()

        # update
        error.append((float(fobj[0]), test_err))
        print(tp.row([idx, float(fobj[0]), test_err]), flush=True)

    return model, Xtest, ytest, yhat, np.array(error)


if __name__ == '__main__':

    model, Xtest, ytest, yhat, error = train(1000, 100, 64, 50, 500, snr=10.)

    # plt.plot(error)
