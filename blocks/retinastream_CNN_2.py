##################################################################
# Building the model
##################################################################
import theano.tensor as T
import numpy as np
import os

from blocks.bricks import Rectifier, MLP
from extrabricks import SoftRectifier
from blocks.bricks.cost import SquaredError
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import Flattener
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.roles import WEIGHT, FILTER
from blocks.graph import ComputationGraph

from metrics import PearsonCorrelation, ExplainedVariance, MeanModelRates, PoissonLogLikelihood

# parameters
batch_size = 256
image_size = 32
pooling_size = 2
num_filters = 1
initial_weight_std = .01
epochs = 5

# filter dimensions
filter_size = 11
num_channels = 40   # filter history

# create theano variables for the stimulus (input) and firing rates (output)
x = T.tensor4('data')
y = T.fcol('rates')

# Build the convolutional Layers
conv_layers = [
    ConvolutionalLayer(Rectifier().apply,
                       (filter_size, filter_size), num_filters,
                       (pooling_size, pooling_size), name='l1')]

convnet = ConvolutionalSequence(
    conv_layers, num_channels=num_channels,
    image_size=(image_size, image_size),
    weights_init=IsotropicGaussian(initial_weight_std),
    biases_init=Constant(0)
    )

convnet.initialize()

output_dim = np.prod(convnet.get_dim('output'))
print(output_dim)

# Fully connected layers
features = Flattener().apply(convnet.apply(x))

mlp = MLP(
        activations=[SoftRectifier()],
        dims=[output_dim, 1],
        weights_init=IsotropicGaussian(initial_weight_std),
        biases_init=Constant(0)
        )
mlp.initialize()

y_hat = mlp.apply(features)


# numerically stable softmax
cost = PoissonLogLikelihood().apply(y.flatten(), y_hat.flatten())
cost.name = 'nll'
mse         = T.mean(SquaredError().cost_matrix(y, y_hat))
mse.name    = 'mean_squared_error'
correlation = PearsonCorrelation().apply(y.flatten(), y_hat.flatten())
explain_var = ExplainedVariance().apply(y.flatten(), y_hat.flatten())
mean_y_hat  = MeanModelRates().apply(y_hat.flatten())

# build the computation graph (CG)
cg = ComputationGraph(cost)

# pick out the weights out of the CG
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
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.algorithms import GradientDescent, Momentum
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model

from retinastream import RetinaStream
from fuel.schemes import SequentialScheme

# set a random seed
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
num_train_examples = 239880     # for 80, 10, 10 split
num_val_examples   = 29985
training_stream    = RetinaStream(filename, datadir, cellidx=1, history=num_channels,
                                  fraction=0.8, seed=seed, partition_label='train',
                                  iteration_scheme=SequentialScheme(num_train_examples, batch_size=batch_size))

validation_stream  = RetinaStream(filename, datadir, cellidx=1, history=num_channels,
                                  fraction=0.8, seed=seed, partition_label='val',
                                  iteration_scheme=SequentialScheme(num_val_examples, batch_size=1024))   # note batch size doesn't matter

model = Model(cost_l2)
algorithm = GradientDescent(
    cost=cost_l2,
    params=model.parameters,
    step_rule=Momentum(
        learning_rate=0.1,
        momentum=0.9)
    )

main_loop = MainLoop(
    model=model,
    data_stream=training_stream,
    algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=epochs),
        TrainingDataMonitoring(
            [cost, correlation, explain_var, mean_y_hat, mse],
            prefix='train',
            after_epoch=True),
        DataStreamMonitoring(
            [cost, correlation, explain_var, mean_y_hat, mse],
            validation_stream,
            prefix='valid'),
        Printing()
        ]
    )

main_loop.run()
