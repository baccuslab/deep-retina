import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM, LSTMMem
from keras.regularizers import l2
from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.optimizers import RMSprop
from keras.objectives import poisson_loss
import h5py
import pickle
import theano
from preprocessing import datagen, loadexpt
from scipy.stats import pearsonr

memories = False #set to true if you want to return the memories (hidden states)
cell_index = 4
numTime = 152
weights_dir = '/home/salamander/Dropbox/deep-retina/saved/lenna.salamander/2015-11-17 11.12.13 lstm/epoch100_iter02300_weights.h5'

def load_Data():
    naturalscenes_test = loadexpt(cell_index, 'naturalscene', 'test', 40)
    X_test = naturalscenes_test.X
    y_test = naturalscenes_test.y
    numTest = (int(X_test.shape[0]/numTime))*numTime
    X_test = X_test[:numTest]
    y_test = y_test[:numTest]
    X_test = np.reshape(X_test, (int(numTest/numTime), numTime, 40, 50, 50))
    y_test = np.reshape(y_test, (int(numTest/numTime), numTime, 1))
    return X_test, y_test

def get_outputs(X_batch):
    l2_reg = 0.01
    stim_shape = (numTime, 40, 50, 50)
    RMSmod = RMSprop(lr=0.001, rho=0.99, epsilon=1e-6)
    num_filters = (8, 16)
    filter_size = (13, 13)
    weight_init = 'he_normal'
    batchsize = 100
    model = Sequential()
    # first convolutional layer
    model.add(TimeDistributedConvolution2D(num_filters[0], filter_size[0], filter_size[1],
                                 input_shape=stim_shape,
                                 border_mode='same', subsample=(1,1),
                                 W_regularizer=l2(l2_reg)))

    #Add relu activation separately for threshold visualizations
    model.add(Activation('relu'))
    # max pooling layer
    model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2), ignore_border=True))

    # flatten
    model.add(TimeDistributedFlatten())

    # Add dense (affine) layer with relu activation
    model.add(TimeDistributedDense(num_filters[1], W_regularizer=l2(l2_reg), activation='relu'))
    # Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
    model.add(LSTM(100*num_filters[1], return_sequences=True))

    # # Add a final dense (affine) layer with softplus activation
    model.add(TimeDistributedDense(1, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))
    model.compile(loss='poisson_loss', optimizer=RMSmod)
    model.load_weights(weights_dir)
    if not memories:
        get_outputs = theano.function([model.layers[0].input], model.layers[5].get_output(train=False))
        outputs = get_outputs(X_batch)
    else:
        model2 = Sequential()
        model2.add(TimeDistributedConvolution2D(num_filters[0], filter_size[0], filter_size[1],
                                 input_shape=stim_shape, weights=model.layers[0].get_weights(),
                                 border_mode='same', subsample=(1,1),
                                 W_regularizer=l2(l2_reg)))

        #Add relu activation separately for threshold visualizations
        model2.add(Activation('relu'))
        # max pooling layer
        model2.add(TimeDistributedMaxPooling2D(pool_size=(2, 2), ignore_border=True))

        # flatten
        model2.add(TimeDistributedFlatten())

        # Add dense (affine) layer with relu activation
        model2.add(TimeDistributedDense(num_filters[1], weights=model.layers[4].get_weights(), W_regularizer=l2(l2_reg), activation='relu'))
        # Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
        model2.add(LSTMMem(100*num_filters[1], weights=model.layers[5].get_weights(), return_memories=True))
        model2.compile(loss='poisson_loss', optimizer=RMSmod)
        get_outputs = theano.function([model2.layers[0].input], model2.layers[5].get_output(train=False))
        outputs = get_outputs(X_batch)
    return outputs

X_test, y_test = load_Data()
outputs = get_outputs(X_test)
print outputs.shape
if not memories:
    pickle.dump(outputs, open("outputs.p", "wb"))
else:
    pickle.dump(outputs, open("memories.p", "wb"))
