import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import LSTMMem #import if using older keras 0.2.0
from keras.regularizers import l2
from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.optimizers import RMSprop, Adam
from keras.objectives import poisson_loss
import h5py
import pickle
import theano
from deepretina.modeltools import load_model, load_partial_model

natural_model_path = './'
natural_weight_name = 'epoch098_iter07000_weights.h5'
natural_multimodel = load_model(natural_model_path, natural_weight_name)
natural_activations = load_partial_model(natural_multimodel, 4)

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : 'first' or 'last', optional
        The axis of the time dimension (default: 'first')

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size `window`.

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

    # flip array dimensinos if the time axis is the first axis
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError("Time axis must be 0 or -1")

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def get_cnn_full_field_flicker(period=1, low_contrast=0.1, high_contrast=1.0):
    sample_rate = 100
    period = int(period)
    flicker_sequence = np.hstack([low_contrast*np.random.randn(period*sample_rate), 
                               high_contrast*np.random.randn(period*sample_rate),
                              low_contrast*np.random.randn(period*sample_rate)])
    
    # Convert flicker sequence into full field movie
    full_field_flicker = np.outer(flicker_sequence, np.ones((1,50,50)))
    full_field_flicker = full_field_flicker.reshape((flicker_sequence.shape[0], 50, 50))

    # Convert movie to 400ms long samples in the correct format for our model
    full_field_movies = rolling_window(full_field_flicker, 40)
    full_field_movies = np.rollaxis(full_field_movies, 2)
    full_field_movies = np.rollaxis(full_field_movies, 3, 1)
    return full_field_movies


data_dir = '/home/salamander/deep-retina-saved-weights/lenna.salamander/2016-01-18 00.48.40 fixedlstm/'
weights_dir = data_dir + 'epoch350_iter01050_weights.h5' #weights for epoch 376 where 0.622 was attained were not saved
nout = len([0, 1, 2, 3, 4])
numTime = 952
l2_reg = 0.01
num_filters = (8, 16)
filter_size = (13, 13)
weight_init = 'he_normal'
batchsize = 100
model = Sequential()

#Add relu activation separately for threshold visualizations
model.add(Activation('relu', input_shape=(numTime, num_filters[1])))

# Add LSTM, forget gate bias automatically initialized to 1, default weight initializations recommended
model.add(LSTMMem(100*num_filters[1], return_sequences=True, return_memories=True))

# # Add a final dense (affine) layer with softplus activation
model.add(TimeDistributedDense(nout, init=weight_init, W_regularizer=l2(l2_reg), activation='softplus'))
model.compile(loss='poisson_loss', optimizer='adam')
model.load_weights(weights_dir)

ntrials = 2000
sample_rate = 100
period = np.ceil((numTime + 40.)/(3*sample_rate))
lstm_responses = np.zeros((ntrials, numTime, 1600)) #since we technically have one sample of 952 timesteps
for n in range(ntrials):
    if n % 50 == 0:
        print('Starting %d trial out of %d.' %(n,ntrials))
    stimulus = get_cnn_full_field_flicker(period=period, low_contrast=0.1, high_contrast=1.0)
    stimulus = stimulus.astype('float32')
    affine_preds = natural_activations(stimulus)
    length = (int(affine_preds.shape[0]/numTime))*numTime #reshape to include timesteps for rnn
    affine_preds = affine_preds[:length]
    rnn_input = np.reshape(affine_preds, (int(length/numTime), numTime, 16))
    get_hiddenstates = theano.function([model.layers[0].input], model.layers[1].get_output(train=False))
    hiddenstates = get_hiddenstates(rnn_input)
    lstm_responses[n, :] = hiddenstates
average_lstm_responses = np.mean(lstm_responses,0)
pickle.dump(average_lstm_responses, open("lstm_hiddenstates.p", "wb"))
