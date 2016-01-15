"""
Toy demo for sequence prediction using LSTM layers
Uses Keras v0.3

code adapted from: http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/

2015/01/12

"""

import numpy as np
import pandas as pd
from random import random
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import matplotlib.pyplot as plt

def generate_data():
    """
    Generates a toy data sequence to train on

    """

    flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
    pdata = pd.DataFrame({"a":flow, "b":flow})
    pdata.b = pdata.b.shift(9)
    return pdata.iloc[10:] * random()  # some noise

def _load_data(data, n_prev = 100):
    """
    helper function

    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':

    # generate sample data
    data = generate_data()
    (X_train, y_train), (X_test, y_test) = train_test_split(data)

    nb_epochs = 10
    batch_size = 1000

    # build the model
    n_hidden = 32
    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(batch_size, 100, 2), stateful=True))
    model.add(Dense(2, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # train
    for epoch in range(nb_epochs):

        print("------------------ Epoch {} of {} ------------------"
              .format(epoch+1, nb_epochs))

        for X, y in datagen(batch_size, X_train, y_train):
            fobj, acc = model.train_on_batch(X, y, accuracy=True)
            print("f = {:2.2f}\ta = {:2.2f}".format(float(fobj), float(acc)))

    # predict
    y_pred = model.predict(X_test)

    # plot
    plt.plot(y_test[:100,:], '-')   # true responses
    plt.plot(y_pred[:100,:], '-')   # predicted responses
    plt.show()
    plt.draw()
