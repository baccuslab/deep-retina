from __future__ import absolute_import

# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l1, l2, activity_l1, activity_l2

def convnet(num_layers=3, filter_sizes=[9], num_filters=[16, 32], pooling_sizes=[2],
        num_channels=40, data_height=50, l2_regularization=0.0):
    '''
    Returns a convolutional neural network model.

    INPUTS
    num_layers          integer; number of layers not including input layer
    
    filter_sizes        list of integers; size of convolutional filters. If
                        len(filter_sizes) < len(num_layers), then remaining
                        layers will be affine.
    
    num_filters         list of integers; number of filters learned per layer.
                        If len(num_filters) < len(num_layers), remaining layers
                        will learn a single filter type.
    
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

    for layer in range(num_layers):
        # if convolutional layer
        if layer < num_convolutional_layers:
            model.add(Convolution2D(num_filters[layer], num_channels,
                filter_sizes[layer], filter_sizes[layer], init='normal',
                border_mode='same', subsample=(1,1),
                W_regularizer=l2(l2_regularization), activation='relu'))
        # if not convolutional, then it's an affine layer
        else:
            model.add(Dense(num_filters[layer-1] * input_height**2,
                num_filters[layer], init='normal',
                W_regularizer=l2(l2_regularization)))

        # if pooling layer follows convolutional or affine layer
        if layer < num_pooling_layers:
            model.add(MaxPooling2D(poolsize=(pooling_sizes[layer], pooling_sizes[layer]),
                ignore_border=True))

        # if it's the last convolutional layer, then need to flatten layer
        if layer == num_convolutional_layers - 1:
            model.add(Flatten())

    # After creating all convolutional, affine, and pooling layers, we're going
    # to add a softplus activation to the top of the model
    model.add(Activation('softplus'))

