# Trains a 3 layer convolutional neural network with the following architecture:
# conv - relu - pool - affine - relu - affine - softplus
#Stimulus is binary white noise 32 x 32 x 40 frames
#Loss: Poisson

from __future__ import absolute_import
import numpy as np
import pickle
from scipy.io import loadmat
import os.path as path
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
# Keras imports
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.embeddings import Embedding
from keras.regularizers import l1, l2, activity_l1, activity_l2
#Imports to add Poisson objective (since Keras does not have them)
import theano
import theano.tensor as T
from six.moves import range

model_basename = 'three_layer_convnet_weights'
num_epochs = 1 #set number of epochs for training

def gaussian(x=np.linspace(-5,5,50),sigma=1.,mu=0.):
     return np.array([(1./(2.*np.pi*sigma**2))*np.exp((-(xi-mu)**2.)/(2.*sigma**2)) for xi in x])

def rolling_window(array, window):
    """
    Make an ndarray with a rolling window of the last dimension
    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def loadData(data_dir):
	metadata = np.load(path.join(data_dir, 'metadata.npz'))['metadata'].item()
	stim  = np.memmap(path.join(data_dir, 'stim_norm.dat'), dtype=metadata['stim_norm_dtype'], 
	                  mode='r', shape=metadata['stim_norm_shape'])
	rates = np.memmap(path.join(data_dir, 'rates.dat'), dtype=metadata['rates_dtype'], 
	                  mode='r', shape=metadata['rates_shape'])
	#Smooth raw spike count with 10 ms std Gaussian to get PSTHs
	rates_filt = np.zeros(rates.shape)
	filt = gaussian(x=np.linspace(-5,5,10), sigma=1, mu=0)
	for cell in xrange(rates.shape[1]):
	    rates_filt[:,cell] = np.convolve(rates[:,cell], filt, mode='same')
	#Create 4d stim array where each data point is a 400ms (40 frame) movie
	stim_sliced = stim[34:-34, 34:-34,:]
	X = rolling_window(stim_sliced, 40)
	X = np.rollaxis(X, 2)
	X = np.rollaxis(X, 3, 1)
	#Truncate rates to appropriate time frame
	y = rates_filt[X.shape[1]:]
	return X, y

def createTrainValTest(X, y, cell):
    # Divide examples into training, validation, and test sets
    # don't need to zero mean data since we loaded stim_norm
    numTrain = 50000
    numVal   = 500
    numTest  = 500

    drawIndices = np.random.choice(X.shape[0], size=(numTrain+numVal+numTest), replace=False)
    trainMask = drawIndices[:numTrain]
    # valMask   = drawIndices[numTrain:-numTest]
    testMask = drawIndices[-numTest:]
    X_train = X[trainMask,:,:,:] #will use validation split to hold out random 500 examples for valset
    y_train = y[trainMask,cell]
    X_test = X[testMask,:,:,:]
    y_test = y[testMask,cell]
    return X_train, y_train, X_test, y_test

def poisson_loss(y_true, y_pred):
    #Negative log likelihood of data y_true given predictions y_pred, according to a Poisson model
    #Assumes that y_pred is > 0

    return T.mean(y_pred - y_true * T.log(y_pred), axis=-1)


def trainNet(X_train, y_train, X_test, y_test):
    model = Sequential()
    #border_mode = full is the default scipy.signal.convolve2d value to do a full linear convolution of input
    #subsample=(1,1) gives a stride of 1
    model.add(Convolution2D(16, 40, 9, 9, init='normal', border_mode='full', subsample=(1,1), W_regularizer=l2(0.0))) 
    model.add(Activation('relu'))
    #ignore_border is the default, since usually not ignoring the border results in weirdness
    model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=True))
    # model.add(Dropout(0.25)) #example of adding dropout

    model.add(Flatten())
    model.add(Dense(6400, 32, init='normal', W_regularizer=l2(0.0)))
    model.add(Activation('relu'))

    model.add(Dense(32, 1, init='normal', W_regularizer=l2(0.0)))
    model.add(Activation('softplus'))
    #Default values (recommended) of RMSprop are learning rate=0.001, rho=0.9, epsilon=1e-6
    #holds out 500 of the 50000 training examples for validation
    model.compile(loss=poisson_loss, optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=50, nb_epoch=num_epochs, verbose=1, validation_split=0.01)
    #saves the weights to HDF5 for potential later use
    model.save_weights(model_basename + str(num_epochs))
    #Would not need accuracy since that is for classification (e.g. F1 score), whereas our problem is regression,
    #so likely we will set show_accuracy=False
    score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=1)
    print('Test score:', score)
    #save test score
    pickle.dump(score, open(model_basename + str(num_epochs) + "_testsetscore.p", "wb"))

print "Loading training data and test data..."
print "(This might take awhile)"
data_dir = '/farmshare/user_data/anayebi/white_noise/'
[X, y] = loadData(data_dir)
cell = 9
[X_train, y_train, X_test, y_test] = createTrainValTest(X, y, cell)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
print "Training and test data loaded. Onto training for " + str(num_epochs) + " epochs..."
trainNet(X_train, y_train, X_test, y_test)
