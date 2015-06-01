# I'll build a convolutional neural network to work on mnist

##################################################################
# Building the model
##################################################################
import pdb
import theano.tensor as T
import numpy as np

from blocks.bricks import Rectifier, Softmax, MLP
from blocks.bricks.cost import SquaredError, MisclassificationRate
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, MaxPooling
from blocks.bricks.conv import ConvolutionalActivation, Flattener
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.roles import WEIGHT, FILTER, INPUT
from blocks.graph import ComputationGraph, apply_dropout

batch_size = 128
filter_size = 3
num_filters = 4
initial_weight_std = .01
epochs = 5

x = T.tensor4('data')
y = T.fvector('rates')

# Convolutional Layers
conv_layers = [
        ConvolutionalLayer(Rectifier().apply, (3,3), 16, (2,2), name='l1'),
        ConvolutionalLayer(Rectifier().apply, (3,3), 32, (2,2), name='l2')]

convnet = ConvolutionalSequence(
        conv_layers, num_channels=40, image_size=(32,32),
        weights_init=IsotropicGaussian(0.1),
        biases_init=Constant(0)
        )

convnet.initialize()

output_dim = np.prod(convnet.get_dim('output'))
print(output_dim)

# Fully connected layers
features = Flattener().apply(convnet.apply(x))

mlp = MLP(
        activations=[Rectifier(), None],
        dims=[output_dim, 100, 1],
        weights_init=IsotropicGaussian(0.01),
        biases_init=Constant(0)
        )
mlp.initialize()

y_hat = mlp.apply(features)


# numerically stable softmax
cost = T.mean(SquaredError().cost_matrix(y.flatten(), y_hat))
cost.name = 'nll'
error_rate = MisclassificationRate().apply(y.flatten(), y_hat)
#cost = MisclassificationRate().apply(y, y_hat)
#cost.name = 'error_rate'

cg = ComputationGraph(cost)

#pdb.set_trace()
weights = VariableFilter(roles=[FILTER, WEIGHT])(cg.variables)
l2_regularization = 0.005 * sum((W**2).sum() for W in weights)

cost_l2 = cost + l2_regularization
cost.name = 'cost_with_regularization'

# Print sizes to check
print("Representation sizes:")
for layer in convnet.layers:
    print(layer.get_dim('input_'))

##################################################################
# Training
##################################################################
from blocks.dump import load_parameter_values
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import SimpleExtension, FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale, Momentum
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, LoadFromDump
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model

#from fuel.datasets import MNIST
from fuel.streams import DataStream
from retinastream import RetinaStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

import os
import h5py

#rng = np.random.RandomState(1)
seed = np.random.randint(100)

# LOAD DATA
machine_name = 'marr'
if machine_name == 'lenna':
    datadir = os.path.expanduser('~/experiments/data/012314b/')
elif machine_name == 'lane':
    datadir = '/Volumes/data/Lane/binary_white_noise/'
elif machine_name == 'marr':
    datadir = os.path.expanduser('~/deepretina/datasets/binary_white_noise/')
filename = 'retina_012314b.hdf5'
print 'Loading RetinaStream'

num_total_examples = 299850
num_train_examples = 239880 # for 80, 10, 10 split
num_val_examples   = 29985
training_stream    = RetinaStream(filename, datadir, cellidx=1, history=40, fraction=0.8, seed=seed,
        partition_label='train',
        iteration_scheme=SequentialScheme(num_train_examples, batch_size=batch_size))
validation_stream  = RetinaStream(filename, datadir, cellidx=1, history=40, fraction=0.8, seed=seed,
        partition_label='val',
        iteration_scheme=SequentialScheme(num_val_examples, batch_size=512))

algorithm = GradientDescent(cost=cost, params=cg.parameters,
        step_rule=Scale(learning_rate=0.1))

model = Model(cost_l2)
algorithm = GradientDescent(
        cost=cost_l2,
        params=model.parameters,
        step_rule=Momentum(
            learning_rate=1e-2,
            momentum=0.9)
        )

main_loop = MainLoop(
        model = model,
        data_stream = training_stream,
        algorithm = algorithm,
        extensions = [
            FinishAfter(after_n_epochs=epochs),
            TrainingDataMonitoring(
                [cost],
                prefix='train',
                after_epoch=True),
            DataStreamMonitoring(
                [cost, error_rate],
                validation_stream,
                prefix='valid'),
            Checkpoint('retinastream_model.pkl', after_epoch=True),
            #EarlyStoppingDump('/Users/jadz/Documents/Micelaneous/Coursework/Blocks/mnist-blocks/', 'valid_error_rate'),
            Printing()
            ]
        )

main_loop.run()


