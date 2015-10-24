# Trains a convolutional + LSTM hybrid neural network with the following architecture:
# conv - relu - pool - affine - relu - LSTM - time distributed affine - softplus
#Stimulus is binary white noise 32 x 32 x 40 frames
#Loss: Poisson
# Requires use of extra layers for Keras: https://github.com/anayebi/keras-extra
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
# from matplotlib.pyplot import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
# Keras imports
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Reshape, Permute
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D
from keras.layers.recurrent import LSTM, GRU, JZS1, JZS2, JZS3
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.embeddings import Embedding
from keras.regularizers import l1, l2, activity_l1, activity_l2
from keras.callbacks import Callback
from keras.objectives import poisson_loss
#Imports to add Poisson objective (since Keras does not have them)
import theano
import theano.tensor as T
# from six.moves import range

model_basename = 'conv-lstm_weights'
num_epochs = 1200 #set number of epochs for training

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
	X = np.rollaxis(X, 3, 1) #need to smooth this? and is it upsampled?
	#Truncate rates to appropriate time frame
	y = np.array(scenes['train/response/firing_rate_10ms'])
	y = y.T
	y = y[40:,:]
	stim_2 = np.array(scenes['test/stimulus'])
	stim_2 = stim_2.T
	X_2 = rolling_window(stim_2, 40)
	X_2 = np.rollaxis(X_2, 2)
	X_2 = np.rollaxis(X_2, 3, 1)
	#Truncate rates to appropriate time frame
	y_2 = np.array(scenes['test/response/firing_rate_10ms'])
	y_2 = y_2.T
	y_2 = y_2[40:,:]
	print X.shape
	print y.shape
	print X_2.shape
	print y_2.shape
	return X, y, X_2, y_2

def createTrainValTest(X, y, X_2, y_2, cell, ts):
	# Divide examples into training, validation, and test sets
	# don't need to zero mean data since we loaded stim_norm

	extent = X.shape[0] #should set this to X.shape[0] when doing full training
	numTime = ts #191 frames, 151 + 40 new frames: ~1.9 seconds
	numTrain = ((extent)/numTime)*numTime
	X_train = X[:numTrain, :, :, :]
	y_train = y[:numTrain, cell]
	extent2 = X_2.shape[0]
	numTest = ((extent2)/numTime)*numTime
	X_test = X_2[:numTest, :, :, :]
	y_test = y_2[:numTest, cell]
	X_train = np.reshape(X_train, (numTrain/numTime, numTime, 40, 50, 50))
	y_train = np.reshape(y_train, (numTrain/numTime, numTime, 1))
	X_test = np.reshape(X_test, (numTest/numTime, numTime, 40, 50, 50))
	y_test = np.reshape(y_test, (numTest/numTime, numTime, 1))
	return X_train, y_train, X_test, y_test

# def poisson_loss(y_true, y_pred):
# 	#Negative log likelihood of data y_true given predictions y_pred, according to a Poisson model
# 	#Assumes that y_pred is > 0

# 	return T.mean(y_pred - y_true * T.log(y_pred), axis=-1)

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		fig = plt.gcf()
		fig.set_size_inches((20,24))
		ax = plt.subplot()
		ax.plot(self.losses, 'b')
		ax.plot(self.losses, 'g')
		ax.set_title('Training loss history', fontsize=16)
		ax.set_xlabel('Iteration', fontsize=14)
		ax.set_ylabel('Training Loss', fontsize=14)

		plt.tight_layout()
		filename = '%dLoss.png' %(num_epochs)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

class ValLossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('val_loss'))

class CorrelationHistory(Callback):
	def on_train_begin(self, logs={}):
		self.train_correlations = []
		self.test_correlations = []

	def on_batch_end(self, batch, logs={}):
		train_subset = range(30) #np.random.choice(X_train.shape[0], 100, replace=False) #around a minute
		test_subset = range(X_test.shape[0]) #np.random.choice(X_test.shape[0], 100, replace=False)
		train_pred = self.model.predict(X_train[train_subset])
		train_pred = train_pred.squeeze()
		test_pred = self.model.predict(X_test[test_subset]) #could change this to all?
		test_pred = test_pred.squeeze()
		train_true = y_train[train_subset].squeeze()
		test_true = y_test[test_subset].squeeze()
		# store just the pearson correlation r averaged over the samples, not the p-value
		train_pred = train_pred.flatten()
		train_true = train_true.flatten()
		test_pred = test_pred.flatten()
		test_true = test_true.flatten()
		self.train_correlations.append(pearsonr(train_pred, train_true)[0])
		self.test_correlations.append(pearsonr(test_pred, test_true)[0])
		fig = plt.gcf()
		fig.set_size_inches((20,24))
		ax = plt.subplot()
		ax.plot(self.train_correlations, 'b')
		ax.plot(self.test_correlations, 'g')
		ax.set_title('Train and Test Pearson Correlations', fontsize=16)
		ax.set_xlabel('Iteration', fontsize=14)
		ax.set_ylabel('Correlation', fontsize=14)

		plt.tight_layout()
		filename = '%dCorrelation.png' %(num_epochs)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

def trainNet(X_train, y_train, X_test, y_test):
	init_method = 'he_uniform' #He et al. paper initializes both conv and fc layers with this
	model = Sequential()
	reg = 0.0 #could try 0.01

	model.add(TimeDistributedConvolution2D(16, 40, 9, 9, init=init_method, border_mode='full', subsample=(1,1), W_regularizer=l1(reg))) 
	model.add(Activation('relu'))

	model.add(TimeDistributedMaxPooling2D(poolsize=(2, 2), ignore_border=True))

	model.add(TimeDistributedFlatten())
	model.add(TimeDistributedDense(13456, 32, init=init_method, W_regularizer=l1(reg)))
	model.add(Activation('relu'))
	#We initialize the bias of the forget gate to be 1, as recommended by Jozefowicz et al.
	model.add(LSTM(32, 32, forget_bias_init='one', return_sequences=True))
	#The predictions of the LSTM network across time given the hidden states
	model.add(TimeDistributedDense(32, 1, init=init_method, activation='softplus', W_regularizer=l1(reg)))

	model.compile(loss='poisson_loss', optimizer='rmsprop')
	history = LossHistory()
	val_history = ValLossHistory()
	corrs = CorrelationHistory()
	checkpointer = ModelCheckpoint(filepath=model_basename+"_bestvallossweights.hdf5", verbose=1, save_best_only=True)
	#stopearly = EarlyStopping(monitor='loss', patience=100, verbose=0)
	model.fit(X_train, y_train, batch_size=50, nb_epoch=num_epochs, verbose=1, validation_data = (X_test, y_test), callbacks=[history, val_history, checkpointer, corrs])
	#saves the weights to HDF5 for potential later use
	model.save_weights(model_basename + str(num_epochs)+".hdf5", overwrite=True)
	#Would not need accuracy since that is for classification (e.g. F1 score), whereas our problem is regression,
	#so likely we will set show_accuracy=False
	#score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=1)
	#print('Test score:', score)
	#save test score
	#pickle.dump(score, open(model_basename + str(num_epochs) + "_testsetscore.p", "wb"))
	
	#Figure to visualize loss history after each batch
	fig = plt.gcf()
	fig.set_size_inches((20,24))
	ax1 = plt.subplot(2,1,1)
	ax1.plot(history.losses, 'k')
	ax1.set_title('Loss history', fontsize=16)
	ax1.set_xlabel('Iteration', fontsize=14)
	ax1.set_ylabel('Loss', fontsize=14)

	ax2 = plt.subplot(2,1,2)
	ax2.plot(corrs.train_correlations, 'b')
	ax2.plot(corrs.test_correlations, 'g')
	ax2.set_title('Train and Test Pearson Correlations', fontsize=16)
	ax2.set_xlabel('Iteration', fontsize=14)
	ax2.set_ylabel('Correlation', fontsize=14)

	plt.tight_layout()
	filename = '%dEpochs.png' %(num_epochs)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	
	pickle.dump(history.losses, open(model_basename + str(num_epochs) + "_losshistory.p", "wb"))
	pickle.dump(val_history.losses, open(model_basename + str(num_epochs) + "_vallosshistory.p", "wb"))

print "Loading training data and test data..."
print "(This might take awhile)"
nat_scene = True
if nat_scene == True:
	print "Training on natural scenes"
	data_dir = '/afs/ir/users/a/n/anayebi/deepRetina/natural-scenes/naturalscene.h5'
else:
	print "Training on white noise"
	data_dir = '/afs/ir/users/a/n/anayebi/deepRetina/natural-scenes/whitenoise.h5'
[X, y, X_2, y_2] = loadData(data_dir)
cell = 0
ts = 152
print "Number of timesteps " + str(ts)
[X_train, y_train, X_test, y_test] = createTrainValTest(X, y, X_2, y_2, cell, ts)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
print "Training and test data loaded for cell " + str(cell) + ". Onto training for " + str(num_epochs) + " epochs..."
trainNet(X_train, y_train, X_test, y_test)
