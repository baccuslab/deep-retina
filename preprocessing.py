from __future__ import absolute_import
import numpy as np
import h5py

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

        # TRAIN DATA
	stim_train = np.array(scenes['train/stimulus'])
	stim_train = stim_train.T
	X_train = rolling_window(stim_train, 40)
	X_train = np.rollaxis(X_train, 2)
	X_train = np.rollaxis(X_train, 3, 1)
	#Truncate rates to appropriate time frame
	y_train = np.array(scenes['train/response/firing_rate_10ms'])
	for i in xrange(y_train.shape[0]): #normalize firing rate of each cell to be between 0 and 1
		if np.max(y_train[i]) != 0:
			y_train[i] /= np.max(y_train[i])
	y_train = y_train.T
	y_train = y_train[40:,:]

        # TEST DATA
	stim_test = np.array(scenes['test/stimulus'])
	stim_test = stim_test.T
	X_test = rolling_window(stim_test, 40)
	X_test = np.rollaxis(X_test, 2)
	X_test = np.rollaxis(X_test, 3, 1)
	#Truncate rates to appropriate time frame
	y_test = np.array(scenes['test/response/firing_rate_10ms'])
	for i in xrange(y_test.shape[0]): #normalize firing rate of each cell to be between 0 and 1
		if np.max(y_test[i]) != 0:
			y_test[i] /= np.max(y_test[i])
	y_test = y_test.T
	y_test = y_test[40:,:]
	print X_train.shape
	print y_train.shape
	print X_test.shape
	print y_test.shape
	return X_train, y_train, X_test, y_test
