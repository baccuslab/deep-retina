import numpy as np
import theano
import theano.tensor as T
import blocks

import os
import h5py

import pyret.spiketools as spkt

from nems.utilities import rolling_window

from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.bricks.conv import Convolutional, ConvolutionalLayer, ConvolutionalActivation
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Rectifier
from blocks.bricks.cost import SquaredError

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
#from fuel.transformers import Flatten
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.plot import Plot

## Load data
mnist = MNIST("train")
train_stream = DataStream.default_stream(mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256))
test_stream  = DataStream.default_stream(mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256))

# MAKE MODEL
# First convolutional layer
convlayer = ConvolutionalLayer(Rectifier().apply, filter_size=(11,11), num_filters=2, num_channels=1, batch_size=256, pooling_size=(10,10), image_size=(32,32), weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
convlayer.initialize()

# initialize data types
#x = T.matrix('features')
x = T.ftensor4('features')
#y = T.lmatrix('targets')
y = T.fvector('targets')
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

