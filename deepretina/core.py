from keras.models import Model
from .glms import GLM
from time import time

__all__ = ['train']


def train(model, data, monitor, num_epochs, reduce_lr_every=-1, reduce_rate=1.0):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model or glms.GLM
        A GLM or Keras Model object

    data : experiments.Experiment
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

    # loop over epochs
    try:
        for epoch in range(num_epochs):
            print('Epoch #{} of {}'.format(epoch + 1, num_epochs))

            # update learning rate on reduce_lr_every, assuming it is positive
            if (reduce_lr_every > 0) and (epoch > 0) and (epoch % reduce_lr_every == 0):
                lr = model.optimizer.lr.get_value()
                model.optimizer.lr.set_value(lr * reduce_rate)
                print('\t(Changed learning rate to {} from {})'.format(lr * reduce_rate, lr))

            # loop over data batches for this epoch
            for X, y in data.train(shuffle=True):

                # update on save_every, assuming it is positive
                if (monitor is not None) and (iteration % monitor.save_every == 0):

                    # performs validation, updates performance plots, saves results to dropbox
                    monitor.save(epoch, iteration, X, y, model.predict)

                # train on the batch
                tstart = time()
                loss = model.train_on_batch(X, y)[0]
                elapsed_time = time() - tstart

                # update
                iteration += 1
                print('[{}]\tLoss: {:5.4f}\t\tIteration time: {:5.2f} seconds'.format(iteration, float(loss), elapsed_time))

    except KeyboardInterrupt:
        print('\nCleaning up')

    print('\nTraining complete!')
    # monitor.cleanup
