from __future__ import absolute_import
import numpy as np
import pickle
from scipy.io import loadmat
import os.path as path
import matplotlib
import h5py

#Force matplotlib to not use any Xwindows else will crash on Rye
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Keras imports
from keras.preprocessing import sequence
import theano
import theano.tensor as T

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
	scenes = h5py.File(data_dir, 'r')
	stim = np.array(scenes['train/stimulus'])
	stim = stim.T
	X = rolling_window(stim, 40)
	X = np.rollaxis(X, 2)
	X = np.rollaxis(X, 3, 1)
	#Truncate rates to appropriate time frame
	y = np.array(scenes['train/response/firing_rate_10ms'])
	for i in xrange(y.shape[0]): #normalize firing rate of each cell to be between 0 and 1
		if np.max(y[i]) != 0:
			y[i] /= np.max(y[i])
	y = y.T
	y = y[40:,:]
	stim_2 = np.array(scenes['test/stimulus'])
	stim_2 = stim_2.T
	X_2 = rolling_window(stim_2, 40)
	X_2 = np.rollaxis(X_2, 2)
	X_2 = np.rollaxis(X_2, 3, 1)
	#Truncate rates to appropriate time frame
	y_2 = np.array(scenes['test/response/firing_rate_10ms'])
	for i in xrange(y_2.shape[0]): #normalize firing rate of each cell to be between 0 and 1
		if np.max(y_2[i]) != 0:
			y_2[i] /= np.max(y_2[i])
	y_2 = y_2.T
	y_2 = y_2[40:,:]
	print X.shape
	print y.shape
	print X_2.shape
	print y_2.shape
	return X, y, X_2, y_2
