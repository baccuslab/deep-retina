import numpy as np
import theano
import theano.tensor as T
import blocks

import os
import h5py

import pyret.spiketools as spkt

from nems.utilities import rolling_window

from blocks.graph import ComputationGraph
from blocks.bricks.conv import Convolutional, ConvolutionalLayer, ConvolutionalActivation
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Rectifier
from blocks.bricks.cost import SquaredError

## Load data
datadir = os.path.expanduser('~/experiments/data/012314b/')
filename = 'retina_012314b.hdf5'
f = h5py.File(os.path.join(datadir, filename))

stim = f['stimulus/sequence']
timestamps = f['stimulus/timestamps']
spk = f['spikes']

# bin spikes
bspk, tbins = spkt.binspikes(spk['cell1'], time=np.append(timestamps, timestamps[-1]+0.01))
bspk = bspk[40:]
rates = spkt.estfr(tbins, bspk, sigma=0.01)

# slice the stimulus
stim_sliced = stim[34:-34, 34:-34, :]

# roll the stimulus
X = rolling_window(stim_sliced, 40)
X = np.rollaxis(X, 2)
X = np.rollaxis(X, 3, 1)
Y = rates[X.shape[1]:]

# MAKE MODEL
# First convolutional layer
convlayer = ConvolutionalLayer(Rectifier().apply, filter_size=(11,11), num_filters=2, num_channels=40, batch_size=256, pooling_size=(10,10), image_size=(32,32), weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
convlayer.initialize()

# initialize data types
x = T.ftensor4('data')
y = T.fvector('rates')
y_hat = convlayer.apply(x)


# SNAP ON THE LOSS FUNCTION
cost = SquaredError().apply(y, y_hat)

# computational graph
cg = ComputationGraph(cost)

# split into Train / Test data
#inds = np.arange(Y.shape[0])
#np.random.shuffle(inds)

#train_inds = inds[:np.round(0.8*inds.size)]
#N = len(train_inds)
