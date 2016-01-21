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
    model.add(LSTM(n_hidden, input_shape=(n_input, 1)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=adam())

    return model


def train(niter, nsteps, nhidden, ntrain, ntest, snr=10.0, disp=True):

    # build a model
    model = build(nsteps, nhidden)

    # generate a test batch
    Xtest, ytest = generate_batch(ntest, nsteps, snr=snr)
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

    # print header
    if disp:
        print(tp.header(['Iter', 'Train', 'Test']))

    error = []

    for idx in range(niter):

        # get data
        X, y = generate_batch(ntrain, nsteps, snr=snr)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # train
        fobj = model.train_on_batch(X, y)

        # test
        yhat = model.predict(Xtest).ravel()
        test_err = ((yhat - ytest) ** 2).mean()

        # update
        error.append((float(fobj[0]), test_err))

        if disp:
            print(tp.row([idx, float(fobj[0]), test_err]), flush=True)

    return model, Xtest, ytest, yhat, np.array(error)


def run_lstm(niter=50, nsteps=100, nhidden=16, ntrain=10, ntest=500, snr=10., disp=True):

    model, Xtest, ytest, yhat, error = train(niter, nsteps, nhidden, ntrain, ntest, snr=snr, disp=disp)
    return model, ytest, yhat, error


def scan_params():
    """
    Scans parameters for training LSTMs

    """

    nh = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
    nt = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    niter = 1000

    data = np.zeros((nh.size, nt.size, niter, 2))

    for i, h in enumerate(nh):
        print('{} of {}'.format(i+1, len(nh)), flush=True)
        for j, t in enumerate(nt):
            model, ytest, yhat, error = run_lstm(niter=niter, nsteps=t, nhidden=h, disp=False, ntest=1000, ntrain=500)
            data[i,j,:,:] = error
            print('\t{} of {}'.format(j+1, len(nt)), flush=True)

    return data, nh, nt


if __name__ == '__main__':
    model, ytest, yhat, error = run_lstm()
