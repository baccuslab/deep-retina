import numpy as np
import theano
import theano.tensor as T
import blocks
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
import pyret.spiketools as spkt

# plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# data handling
import os
import h5py
from nems.utilities import rolling_window
from retinastream import RetinaStream

# blocks
from blocks.bricks.conv import Convolutional, ConvolutionalLayer, ConvolutionalActivation
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Rectifier
from blocks.bricks.cost import SquaredError
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
# from blocks.algorithms import RMSProp
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.plot import Plot

# LOAD DATA
machine_name = 'lane'
if machine_name == 'lenna':
    datadir = os.path.expanduser('~/experiments/data/012314b/')
elif machine_name == 'lane':
    datadir = '/Volumes/data/Lane/binary_white_noise/'
elif machine_name == 'marr':
    datadir = os.path.expanduser('~/retina/deepretina/datasets/binary_white_noise/')
filename = 'retina_012314b.hdf5'
print 'Loading RetinaStream'
train_stream = RetinaStream(filename, datadir, cellidx=1, history=40, fraction=0.8)
test_stream  = RetinaStream(filename, datadir, cellidx=1, history=40, fraction=0.2)

# MAKE MODEL
# First convolutional layer
print 'Initializing ConvLayer'
convlayer = ConvolutionalLayer(Rectifier().apply, filter_size=(11,11), num_filters=2, num_channels=40, batch_size=256, pooling_size=(10,10), image_size=(32,32), weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
convlayer.initialize()

x = T.ftensor4('data')
y = T.fvector('rates')
y_hat = convlayer.apply(x)


# SNAP ON THE LOSS FUNCTION
print 'Initializing Cost'
cost = SquaredError().apply(y_hat, y)

print 'Initializing Computation Graph'
cg = ComputationGraph(cost)
monitor = DataStreamMonitoring(variables=[cost], data_stream=test_stream, prefix="test")

print 'Starting Main Loop'
main_loop = MainLoop(
        model=None, data_stream=train_stream,
        algorithm=GradientDescent(cost=cost, params=cg.parameters,
            step_rule=Scale(learning_rate=0.1)),
        extensions=[FinishAfter(after_n_epochs=1),
            TrainingDataMonitoring([cost], after_batch=True),
            Plot('Plotting example', channels=[['cost']],
                after_batch=True)])

main_loop.run()

