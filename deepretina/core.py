"""
Core tools for training models
"""
from time import time
import tableprint as tp
import keras.backend as K

__all__ = ['train']


def train(model, experiment, monitor, num_epochs):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model or glms.GLM
        A GLM or Keras Model object

    experiment : experiments.Experiment
        An Experiment object

    monitor : io.Monitor
        Saves the model parameters and plots of performance progress

    num_epochs : int or iterable
        Number of epochs to train for
    """
    # initialize training iteration
    iteration = 0
    train_start = time()
    if isinstance(num_epochs, int):
        nepochs = num_epochs
    else:
        nepochs = sum(num_epochs)

    # loop over epochs
    try:
        for epoch in range(nepochs):
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
                loss = model.train_on_batch(X[np.newaxis], y[np.newaxis])
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
