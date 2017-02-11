"""
Core tools for training models
"""
from keras.models import Model
from .glms import GLM
from time import time
import tableprint as tp

__all__ = ['train']


def train(model, experiment, monitor, num_epochs, augment=False):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model or glms.GLM
        A GLM or Keras Model object

    experiment : experiments.Experiment
        An Experiment object

    monitor : io.Monitor
        Saves the model parameters and plots of performance progress

    num_epochs : int
        Number of epochs to train for

    reduce_lr_every : int
        How often to reduce the learning rate

    reduce_rate : float
        A fraction (constant) to multiply the learning rate by

    """
    assert isinstance(model, (Model, GLM)), "'model' must be a GLM or Keras model"

    # initialize training iteration
    iteration = 0
    train_start = time()

    # loop over epochs
    try:
        for epoch in range(num_epochs):
            tp.banner('Epoch #{} of {}'.format(epoch + 1, num_epochs))
            print(tp.header(["Iteration", "Loss", "Runtime"]), flush=True)

            # loop over data batches for this epoch
            for X, y in experiment.train(shuffle=True):

                # update on save_every, assuming it is positive
                if (monitor is not None) and (iteration % monitor.save_every == 0):

                    # performs validation, updates performance plots, saves results to dropbox
                    monitor.save(epoch, iteration, X, y, model.predict)

                # train on the batch
                tstart = time()
                loss = model.train_on_batch({'stim':X, 'loss':y})[0]
                elapsed_time = time() - tstart

                # update
                iteration += 1
                print(tp.row([iteration, float(loss), tp.humantime(elapsed_time)]), flush=True)

            print(tp.bottom(3))

    except KeyboardInterrupt:
        print('\nCleaning up')

    # allows the monitor to perform any post-training visualization
    if monitor is not None:
        elapsed_time = time() - train_start
        monitor.cleanup(iteration, elapsed_time)

    tp.banner('Training complete!')
