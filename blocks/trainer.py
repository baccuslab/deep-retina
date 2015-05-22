import numpy as np
import theano
import theano.tensor as T
import blocks

from blocks.graph import ComputationGraph
from fuel.streams import DataStream
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot

#import layers
#from scipy.stats.stats import pearsonr
#import matplotlib.pyplot as plt
#from os.path import expanduser


class Trainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, Y, train_inds, val_inds, model, loss_function, 
                   reg=0.0, dropout=1.0, learning_rate=1e-2, momentum=0, 
                   learning_rate_decay=0.95, update='momentum', 
                   sample_batches=True, num_epochs=30, batch_size=100, 
                   acc_frequency=None, augment_fn=None, predict_fn=None,
                   verbose=False, save_plots=False, machine='LaneMacbook'):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: 4D array of all data; shape (num_samples, num_channels, img_width, img_height)
    - y: Vector of all labels; y[i] gives the label for X[i].
    - train_inds: vector mask for training data
    - test_inds: vector mask for test data
    - model: A blocks model that has already been initialized 
    - loss_function: A blocks.bricks.cost function of (y, y_hat)
    - reg: Regularization strength. This will be passed to the loss function.
    - dropout: Amount of dropout to use. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - num_epochs: The number of epochs to take over the training data.
    - batch_size: The number of training samples to use at each iteration.
    - acc_frequency: If set to an integer, we compute the training and
      validation set accuracy after every acc_frequency iterations.
    - augment_fn: A function to perform data augmentation. If this is not
      None, then during training each minibatch will be passed through this
      before being passed as input to the network.
    - predict_fn: A function to mutate data at prediction time. If this is not
      None, then during each testing each minibatch will be passed through this
      before being passed as input to the network.
    - verbose: If True, print status after each epoch.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """

    x = T.ftensor4('stimulus')
    y = T.fvector('rates')
    y_hat = model.apply(x)

    cost = loss_function.apply(y, y_hat)

    sample_batches = model.batch_size
    N = Y[train_inds].shape[0] #X.shape[0]
    N_test = Y[test_inds].shape[0]

    train_stream = DataStream(X[train_inds], 
            iteration_scheme=SequentialScheme(N, sample_batches))
    test_stream  = DataStream(X[test_inds],
            iteration_scheme=SequentialScheme(N_test, sample_batches))

    cg = ComputationGraph(cost)

    main_loop = MainLoop(
            model=model, data_stream=train_stream,
            algorithm=update,
            extensions=[FinishAfter(after_n_epochs=num_epochs),
                TrainingDataMonitoring([cost, *params], after_batch=True),
                Plot('Plotting example', channels=[['cost']], after_batch=True)])

    main_loop.run()

    # return the best model and the training history statistics
    return model
