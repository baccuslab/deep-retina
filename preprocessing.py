from __future__ import absolute_import
import numpy as np
import h5py

def rolling_window(array, window):
    """
    Make an ndarray with a rolling window of the last dimension
    Parameters
    ----------
    array : array_like
            Array to add rolling window to
    window : int
            Size of rolling window
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
               [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
               [ 6.,  7.,  8.]])
    """
    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def load_data(data_dir, cell, model_type, ts=None):
    '''
    Returns training and test data formatted as (samples, num_channels, height, width).

    INPUTS
    data_dir            full path to data
    cell                integer of neuron to train/test on
    model_type          'lstm' or 'convnet'
    ts                  number of time steps for LSTM

    OUTPUS
    X_train
    y_train
    X_test
    y_test
    '''
    scenes = h5py.File(data_dir, 'r')

    # TRAIN DATA
    stim_train = np.array(scenes['train/stimulus'])
    stim_train = stim_train.T
    X_train = rolling_window(stim_train, 40)
    X_train = np.rollaxis(X_train, 2)
    X_train = np.rollaxis(X_train, 3, 1)
    #Truncate rates to appropriate time frame
    y_train = np.array(scenes['train/response/firing_rate_10ms'])
    for i in xrange(y_train.shape[0]): #normalize firing rate of each cell to be between 0 and 1
            if np.max(y_train[i]) != 0:
                    y_train[i] /= np.max(y_train[i])
    y_train = y_train.T
    y_train = y_train[40:,cell]

    # TEST DATA
    stim_test = np.array(scenes['test/stimulus'])
    stim_test = stim_test.T
    X_test = rolling_window(stim_test, 40)
    X_test = np.rollaxis(X_test, 2)
    X_test = np.rollaxis(X_test, 3, 1)
    #Truncate rates to appropriate time frame
    y_test = np.array(scenes['test/response/firing_rate_10ms'])
    for i in xrange(y_test.shape[0]): #normalize firing rate of each cell to be between 0 and 1
            if np.max(y_test[i]) != 0:
                    y_test[i] /= np.max(y_test[i])
    y_test = y_test.T
    y_test = y_test[40:,cell]

    if model_type == 'lstm':
        # reshape data for LSTM
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        num_channels = X_train.shape[1]
        height = X_train.shape[2]
        X_train = np.reshape(X_train, (num_train/ts, ts, num_channels, height, height))
        y_train = np.reshape(y_train, (num_train/ts, ts, 1))
        X_test = np.reshape(X_test, (num_test/ts, ts, num_channels, height, height))
        y_test = np.reshape(y_test, (num_test/ts, ts, 1))

    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    return X_train, y_train, X_test, y_test

def create_data_split(X_train, y_train, X_test, y_test, split=0.1,
        train_subset=1.0, test_subset=1.0, batch_size=128):
    '''
    Divide examples into training, validation, and test sets.

    INPUTS
    X_train             X_train from load_data
    y_train             y_train from load_data
    X_test              X_test from load_data
    y_test              y_test from load_data
    split               number of validation examples / number of training examples
    train_subset        fraction of total training data to use
    test_subset         fraction of total test data to use
    batch_size          batch size

    OUTPUS
    train_inds          indices for train batches
    val_inds            indices for validation batches
    test_inds           indices for test batches
    '''

    num_train_val = int(X_train.shape[0] * train_subset)
    num_val = int(split * num_train_val)
    num_train = num_train_val - num_val
    num_test  = int(X_test.shape[0] * test_subset)

    # train and val indices
    draw_indices = np.random.choice(num_train_val, size=(num_train+num_val), replace=False)
    train_mask = draw_indices[:num_train]
    val_mask = draw_indices[num_train:]

    # test indices
    test_mask = np.random.choice(X_test.shape[0], size=num_test, replace=False)

    # generate batches
    train_inds = [train_mask[i:i+batch_size] for i in range(0, len(train_mask), batch_size)]
    val_inds = [val_mask[i:i+batch_size] for i in range(0, len(val_mask), batch_size)]
    test_inds = [test_mask[i:i+batch_size] for i in range(0, len(test_mask), batch_size)]

    return train_inds, val_inds, test_inds

