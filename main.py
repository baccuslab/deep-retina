from __future__ import absolute_import

from keras.optimizers import RMSprop
from keras.objectives import poisson_loss

from preprocessing import load_data
from models import convnet, lstm


# ------------------ Load Data ---------------------- #
data_options = {
    'cell_index': 1,          # which cell to use
    'val_split': 0.01,        # fraction of training data to use for validation
    'ntrain_samples': 1000,   # number of training samples to use
    'ntest_samples': 1000,    # number of test samples to use
    'filter_history': 40,     # number of steps to use in the filter history
    'batch_size': 128         # number of samples to use in each batch
}

train, test, val = load_data('2015-10-07', 'naturalscene', **data_options)
# --------------------------------------------------- #


# --------------- Initialize Model ------------------ #
model_options = {
    'nlayers': 3,
    'filter_sizes': [4],
    'num_filters': [16, 32, 1],
    'pooling_sizes': [2],
    'lstm_timesteps': 100,
    'loss': poisson_loss,
}

model = convnet(trian, test, val, **model_options)
# --------------------------------------------------- #


# -------------- Initialize Optimizer --------------- #
optimizer_options = {
    'lr': 5e-5,             # learning rate
    'rho': 0.99,            # decay rate
    'epsilon': 1e-6         # ???
}

optimizer = RMSprop(**optimizer_options)
# --------------------------------------------------- #


# ------------------ Train Model -------------------- #
training_options = {
    'save_every': 100,      # save every n iterations
    'filename': '',         # where to save results
}

model.train(**training_options)
# --------------------------------------------------- #


# # Initialize model
# if model_type == 'convnet':
    # model = convnet(num_layers=nlayers, filter_sizes=filter_sizes,
            # num_filters=num_filters, pooling_sizes=pooling_sizes,
            # num_channels=nchannels, data_height=X_train.shape[-1],
            # l2_regularization=l2_reg)
# elif model_type == 'lstm':
    # model = lstm(num_layers=nlayers, filter_sizes=filter_sizes,
            # num_filters=num_filters, pooling_sizes=pooling_sizes,
            # num_channels=nchannels, data_height=X_train.shape[-1],
            # l2_regularization=l2_reg)

# Choose optimizer and loss function
# rmsprop = RMSprop(lr=learning_rate, rho=decay_rate, epsilon=1e-6)
# model.compile(loss=poisson_loss, optimizer='rmsprop')

# print('Model compiled. Onto training for %d epochs.' %nepochs)

# Initialize metrics and callbacks
# losses = []
# metrics = {}

# # Train model
# for epoch_idx in range(nepochs):
    # for batch_idx, batch in enumerate(train_inds):
        # new_loss = model.train_on_batch(X_train[batch].astype(float), y_train[batch], accuracy=False)
        # losses.append(new_loss)

        # if batch_idx % check_every == 0:
            # #random_val_batch = np.random.choice(len(val_inds), size=1)
            # #train_preds = model.predict(X_train[batch])
            # #val_preds = model.predict(X_train[val_inds[random_val_batch]])
            # metrics = training_metrics(model, X_train, y_train, batch_size,
                    # batch_size, train_inds, val_inds, metrics)
            # metrics['train_losses'] = losses
            # plot_metrics(metrics, batch_idx)

            # # save model weights
            # weight_filename = 'trained_weights_epoch%d_batch%d.h5' %(epoch_idx, batch_idx)
            # model.save_weights(weight_filename, overwrite=True)

        # print('Finished batch %d in epoch %d. Loss: %f.' %(batch_idx, epoch_idx, new_loss))
