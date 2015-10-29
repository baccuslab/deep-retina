from __future__ import absolute_import

# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l1, l2, activity_l1, activity_l2

def convnet(num_layers=3, filter_sizes=[9], num_filters=[16, 32, 1], pooling_sizes=[2],
        num_channels=40, data_height=50, l2_regularization=0.0):
    '''
    Returns a convolutional neural network model.

    INPUTS
    num_layers          integer; number of layers not including input layer
    
    filter_sizes        list of integers; size of convolutional filters. If
                        len(filter_sizes) < len(num_layers), then remaining
                        layers will be affine.
    
    num_filters         list of integers; number of filters learned per layer.
                        Should have len(num_filters) = len(num_layers).
    
    pooling_sizes       list of integers; square root of units pooled.
                        len(pooling_size) = number of pooling layers.

    num_channels        integer; X.shape[1]; number of time channels in data

    data_height         integer; X.shape[2]; number of pixels on one side of the data

    l2_regularization   float >= 0; strength of L2 regularization

    OUTPUT
    model               keras model
    '''
    
    # Determine architecture
    num_convolutional_layers = len(filter_sizes)
    num_pooling_layers = len(pooling_sizes)
    input_height = data_height

    # Initialize Feedforward Convnet
    model = Sequential()

    # How should we initialize weights?
    init_method = 'normal'

    for layer in range(num_layers):
        # if convolutional layer
        if layer < num_convolutional_layers:
            model.add(Convolution2D(num_filters[layer], num_channels,
                filter_sizes[layer], filter_sizes[layer], init='normal',
                border_mode='same', subsample=(1,1),
                W_regularizer=l2(l2_regularization), activation='relu'))
        # if not convolutional, then it's an affine layer
        else:
            # if not last layer, add relu activation to affine layer
            if layer != num_layers - 1:
                model.add(Dense(num_filters[layer-1] * input_height**2,
                    num_filters[layer], init=init_method,
                    W_regularizer=l2(l2_regularization), activation='relu'))
                # need to set input_height to 1, since now input is flattened
                # after this affine layer
                input_height = 1
            # else it's the last layer and we will add a softplus activation
            else:
                model.add(Dense(num_filters[layer-1] * input_height**2,
                    num_filters[layer], init=init_method,
                    W_regularizer=l2(l2_regularization)))

        # if pooling layer follows convolutional or affine layer
        if layer < num_pooling_layers:
            model.add(MaxPooling2D(poolsize=(pooling_sizes[layer], pooling_sizes[layer]),
                ignore_border=True))
            input_height = input_height // 2 

        # if it's the last convolutional layer, then need to flatten layer
        if layer == num_convolutional_layers - 1:
            model.add(Flatten())

    # After creating all convolutional, affine, and pooling layers, we're going
    # to add a softplus activation to the top of the model
    model.add(Activation('softplus'))

    return model



def lstm(num_layers=3, filter_sizes=[9], num_filters=[16, 32], pooling_sizes=[2],
        num_channels=40, data_height=50, l2_regularization=0.0):
    '''
    Returns a long-short-term-memory (LSTM) network model.

    INPUTS
    num_layers          integer; number of layers not including input layer
    
    filter_sizes        list of integers; size of convolutional filters. If
                        len(filter_sizes) < len(num_layers), then remaining
                        layers will be affine.
    
    num_filters         list of integers; number of filters learned per layer.
                        Should have len(num_filters) = len(num_layers).
    
    pooling_sizes       list of integers; square root of units pooled.
                        len(pooling_size) = number of pooling layers.

    num_channels        integer; X.shape[1]; number of time channels in data

    data_height         integer; X.shape[2]; number of pixels on one side of the data

    l2_regularization   float >= 0; strength of L2 regularization

    OUTPUT
    model               keras model
    '''

    # Determine architecture
    num_convolutional_layers = len(filter_sizes)
    num_pooling_layers = len(pooling_sizes)
    input_height = data_height

    # Fixed parameters
    init_method = 'he_uniform'

    # Initialize Feedforward Convnet
    model = Sequential()

    for layer in range(num_layers)-1:
        # if convolutional layer
        if layer < num_convolutional_layers:
            model.add(TimeDistributedConvolution2D(num_filters[layer], num_channels,
                filter_sizes[layer], filter_sizes[layer], init=init_method,
                border_mode='full', subsample=(1,1),
                W_regularizer=l2(l2_regularization), activation='relu'))
        # if not convolutional, then it's an affine layer
        else:
            # if not last layer, add relu activation to affine layer
            if layer != num_layers - 1:
                model.add(TimeDistributedDense(num_filters[layer-1] * input_height**2,
                    num_filters[layer], init=init_method,
                    W_regularizer=l2(l2_regularization), activation='relu'))


        # if pooling layer follows convolutional or affine layer
        if layer < num_pooling_layers:
            model.add(TimeDistributedMaxPooling2D(poolsize=(pooling_sizes[layer], pooling_sizes[layer]),
                ignore_border=True))
            input_height = input_height // 2 

        # if it's the last convolutional layer, then need to flatten layer
        if layer == num_convolutional_layers - 1:
            model.add(TimeDistributedFlatten())

    model.add(LSTM(num_filters[-1], num_filters[-1], forget_bias_init='one',
        return_sequences=True))

    model.add(TimeDistributedDense(num_filters[-1], 1, init=init_method,
        W_regularizer=l2(l2_regularization), activation='softplus'))

    return model
