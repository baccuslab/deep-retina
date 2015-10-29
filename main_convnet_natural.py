from __future__ import absolute_import
import numpy as np
import os.path as path

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss


from preprocessing import *
from models import *
from callbacks import *


# Which cell to train on?
cell = 1

# Constants
data_dir = '/Volumes/group/baccus/Lane/2015-10-07/naturalscene.h5'
batch_size = 128
val_split = 0.01 # fraction of training data used for validation
train_subset = 0.1 # subset of training data used
test_subset = 1.0 # subset of test data used
nchannels = 40 # time duration of filters learned
check_every = 100 # monitor training after every N batches
nepochs = 1

# Hyperparameters
learning_rate = 5e-5
decay_rate = 0.99
l2_reg = 0.0

# Architecture
model_type = 'convnet' # 'convnet' or 'lstm'
ts = 100 # if lstm
nlayers = 3
filter_sizes = [9]
num_filters = [16, 32]
pooling_sizes = [2]

# Load All Data
X_train, y_train, X_test, y_test = load_data(data_dir, cell, model_type, ts)

# Split Data into Training, Validation, and Test sets.
# This returns the indices into X_train, y_train, X_test, and y_test for each batch.
train_inds, val_inds, test_inds = create_data_split(X_train, y_train, X_test,
        y_test, split=val_split, train_subset=train_subset,
        test_subset=test_subset, batch_size=batch_size)

print('Training and test data loaded. Onto initializing model.')

# Initialize model
if model_type == 'convnet':
    model = convnet(num_layers=nlayers, filter_sizes=filter_sizes,
            num_filters=num_filters, pooling_sizes=pooling_sizes,
            num_channels=nchannels, data_height=X_train.shape[-1],
            l2_regularization=l2_reg)
elif model_type == 'lstm':
    model = lstm(num_layers=nlayers, filter_sizes=filter_sizes,
            num_filters=num_filters, pooling_sizes=pooling_sizes,
            num_channels=nchannels, data_height=X_train.shape[-1],
            l2_regularization=l2_reg)

# Choose optimizer and loss function
rmsprop = RMSprop(lr=learning_rate, rho=decay_rate, epsilon=1e-6)
model.compile(loss=poisson_loss, optimizer='rmsprop')

print('Model compiled. Onto training for %d epochs.' %nepochs)

# Initialize metrics and callbacks
losses = []
metrics = {}

# Train model
for epoch_idx in range(nepochs):
    for batch_idx, batch in enumerate(train_inds):
        new_loss = model.train_on_batch(X_train[batch], y_train[batch], accuracy=False)
        losses.append(new_loss)

        if batch_idx % check_every == 0:
            #random_val_batch = np.random.choice(len(val_inds), size=1)
            #train_preds = model.predict(X_train[batch])
            #val_preds = model.predict(X_train[val_inds[random_val_batch]])
            metrics = training_metrics(model, X_train, y_train, batch_size,
                    batch_size, train_inds, val_inds, metrics)
            metrics['train_losses'] = losses
            plot_metrics(metrics, batch_idx)
            
            # save model weights
            weight_filename = 'trained_weights_epoch%d_batch%d.h5' %(epoch_idx, batch_idx)
            model.save_weights(weight_filename, overwrite=True)

        print('Finished batch %d in epoch %d. Loss: %f.' %(batch_idx, epoch_idx, new_loss)
