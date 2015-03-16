import numpy as np
import layers
from scipy.stats.stats import pearsonr


class ClassifierTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, y, X_val, y_val, 
            model, loss_function, 
            reg=0.0, dropout=1.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            update='momentum', sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            augment_fn=None, predict_fn=None,
            verbose=False):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of training data; each X[i] is a training sample.
    - y: Vector of training labels; y[i] gives the label for X[i].
    - X_val: Array of validation data
    - y_val: Vector of validation labels
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - dropout: Amount of dropout to use. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
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

    N = X.shape[0]

    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0 # if you switch back to error, this needs to be np.inf
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      if verbose:
        if it % 10 == 0:  print 'starting iteration ', it

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y

      # Maybe perform data augmentation
      if augment_fn is not None:
        X_batch = augment_fn(X_batch)

      # evaluate cost and gradient
      cost, grads = loss_function(X_batch, model, y_batch, reg=reg, dropout=dropout)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          dx = momentum * self.step_cache[p] - learning_rate * grads[p]
          self.step_cache[p] = dx
        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          self.step_cache[p] = self.step_cache[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + 1e-8)
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y
        # Computing a forward pass with a batch size of 1000 will is no good,
        # so we batch it
        if sample_batches:
            iterations = X_train_subset.shape[0] / batch_size

            scores       = np.zeros(y_train_subset.shape)
            y_pred_train = np.zeros(y_train_subset.shape)
            for it in xrange(iterations):
                batch_mask = np.random.choice(X_train_subset.shape[0], batch_size)
                X_batch    = X_train_subset[batch_mask]
                y_batch    = y_train_subset[batch_mask]
                y_pred_train[it*batch_size:(it+1)*batch_size] = loss_function(X_batch, model).squeeze()
        else:
            y_pred_train = loss_function(X_train_subset, model) # calling loss_function with y=None returns rates
        

        train_acc, _ = pearsonr(y_pred_train, y_train_subset) 
        train_acc_history.append(train_acc)

        # evaluate val accuracy, but split the validation set into batches
        if sample_batches:
            iterations = X_val.shape[0] / batch_size

            scores     = np.zeros(y_val.shape)
            y_pred_val = np.zeros(y_val.shape)
            for it in xrange(iterations):
                batch_mask  = np.random.choice(X_val.shape[0], batch_size)
                X_val_batch = X_val[batch_mask]
                y_val_batch = y_val[batch_mask]
                y_pred_val[it*batch_size:(it+1)*batch_size] = loss_function(X_val_batch, model).squeeze()
        else:
            y_pred_val = loss_function(X_val, model) # calling loss_function with y=None returns rates


        val_acc, _ = pearsonr(y_pred_val, y_val)
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history


  def train_memmap(self, X, y, train_inds, val_inds, model, loss_function, 
                   reg=0.0, dropout=1.0, learning_rate=1e-2, momentum=0, 
                   learning_rate_decay=0.95, update='momentum', 
                   sample_batches=True, num_epochs=30, batch_size=100, 
                   acc_frequency=None, augment_fn=None, predict_fn=None,
                   verbose=False):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of all data; each X[i] is a training sample.
    - y: Vector of all labels; y[i] gives the label for X[i].
    - train_inds: vector mask for training data
    - test_inds: vector mask for test data
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - dropout: Amount of dropout to use. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
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

    N = y[train_inds].shape[0] #X.shape[0]

    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0 # if you switch back to error, this needs to be np.inf
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      if verbose:
        if it % 10 == 0:  print 'starting iteration ', it

      # get batch of data
      if sample_batches:
        batch_mask = train_inds[np.random.choice(N, batch_size, replace=False)]
      else:
        # no SGD used, full gradient descent
        batch_mask = train_inds

      # Maybe perform data augmentation
      if augment_fn is not None:
        X_batch = augment_fn(X[batch_mask])

      # evaluate cost and gradient
      cost, grads = loss_function(X[batch_mask], model, y[batch_mask], reg=reg, dropout=dropout)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = momentum * self.step_cache[p] - learning_rate * grads[p]
          self.step_cache[p] = dx
        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          self.step_cache[p] = self.step_cache[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + 1e-8)
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = train_inds[np.random.choice(N, 1000, replace=False)]
        else:
          train_mask = train_inds

        # Computing a forward pass with a batch size of 1000 will is no good,
        # so we batch it
        if sample_batches:
            assert y[train_mask].shape[0] % batch_size == 0, 'Size of training data must be divisible by %d.' %(batch_size)
            iterations = y[train_mask].shape[0] / batch_size

            scores       = np.zeros(train_mask.shape)
            y_pred_train = np.zeros(train_mask.shape)
            for it in xrange(iterations):
                batch_mask = train_mask[np.random.choice(y[train_mask].shape[0], batch_size, replace=False)]
                y_pred_train[it*batch_size:(it+1)*batch_size] = loss_function(X[train_mask], model).squeeze()
        else:
            y_pred_train = loss_function(X[train_mask], model) # calling loss_function with y=None returns rates
        

        train_acc, _ = pearsonr(y_pred_train, y[train_mask]) 
        train_acc_history.append(train_acc)

        # evaluate val accuracy, but split the validation set into batches
        if sample_batches:
            assert y[val_inds].shape[0] % batch_size == 0, 'Size of val data must be divisible by %d.' %(batch_size)
            iterations = y[val_inds].shape[0] / batch_size

            scores     = np.zeros(y[val_inds].shape)
            y_pred_val = np.zeros(y[val_inds].shape)
            for it in xrange(iterations):
                batch_mask  = val_inds[np.random.choice(y[val_inds].shape[0], batch_size, replace=False)]
                y_pred_val[it*batch_size:(it+1)*batch_size] = loss_function(X[batch_mask], model).squeeze()
        else:
            y_pred_val = loss_function(X[val_inds], model) # calling loss_function with y=None returns rates


        val_acc, _ = pearsonr(y_pred_val, y[val_inds])
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history




