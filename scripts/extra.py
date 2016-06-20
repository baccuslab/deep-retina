# Extra Layers that I have added to Keras
# Layers that have been added to the Keras master branch will be noted both in the ReadMe and removed from extra.py.
#
# Copyright Aran Nayebi, 2015
# anayebi@stanford.edu
#
# If you already have Keras installed, for this to work on your current installation, please do the following:
# 1. Upgrade to the newest version of Keras (since some layers may have been added from here that are now commented out):
#    sudo pip install --upgrade git+git://github.com/fchollet/keras.git
# or, if you don't have super user access, just run:
#    pip install --upgrade git+git://github.com/fchollet/keras.git --user
#
# 2. Add this file to your Keras installation in the layers directory (keras/layers/)
#
# 3. Now, to use any layer, just run:
#    from keras.layers.extra import layername
#
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import theano.tensor as T
from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import Recurrent

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

class ImgReshape(Layer):
    # This layer reshapes an input of shape (samples, time, dim) to (samples, time, num_subunits, num_rows, num_cols)
    input_ndim = 3
    def __init__(self, nb_subunits, **kwargs):
        self.nb_subunits = nb_subunits
        self.input = K.placeholder(ndim=3)
        super(ImgReshape, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        num_units = input_shape[2]
        num_rows = (int)(np.sqrt(num_units/self.nb_subunits))
        return (input_shape[0], input_shape[1], self.nb_subunits, num_rows, num_rows)

    def get_output(self, train=False):
        X = self.get_input(train)
        input_shape = self.input_shape
        num_units = input_shape[2]
        num_rows = (int)(np.sqrt(num_units/self.nb_subunits))
        Y = K.reshape(X, (X.shape[0], X.shape[1], self.nb_subunits, num_rows, num_rows))
        return Y

class ConvRNN(Recurrent):
    """RNN with all connections being convolutions:
    H_t = activation(conv(H_tm1, W_hh) + conv(X_t, W_ih) + b)
    with H_t and X_t being images and W being filters.
    We use Keras' RNN API, thus input and outputs should be 3-way tensors.
    Assuming that your input video have frames of size
    [nb_channels, nb_rows, nb_cols], the input of this layer should be reshaped
    to [batch_size, time_length, nb_channels*nb_rows*nb_cols]. Thus, you have to
    pass the original images shape to the ConvRNN layer.
    Parameters:
    -----------
    filter_dim: list [nb_filters, nb_row, nb_col] convolutional filter
        dimensions
    reshape_dim: list [nb_channels, nb_row, nb_col] original dimensions of a
        frame.
    batch_size: int, batch_size is useful for TensorFlow backend.
    time_length: int, optional for Theano, mandatory for TensorFlow
    subsample: (int, int), just keras.layers.Convolutional2D.subsample
    """
    def __init__(self, filter_dim, reshape_dim,
                 batch_size=None, subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, **kwargs):
        self.batch_size = batch_size
        self.border_mode = 'same'
        self.filter_dim = filter_dim
        self.reshape_dim = reshape_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.initial_weights = weights

        self.subsample = tuple(subsample)
        self.output_dim = (filter_dim[0], reshape_dim[1]//self.subsample[0],
                           reshape_dim[2]//self.subsample[1])

        super(ConvRNN, self).__init__(**kwargs)

    def _get_batch_size(self, X):
        if K._BACKEND == 'theano':
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        return batch_size

    def build(self):
        if K._BACKEND == 'theano':
            batch_size = None
        else:
            batch_size = None  # self.batch_size
        input_dim = self.input_shape
        bm = self.border_mode
        reshape_dim = self.reshape_dim
        hidden_dim = self.output_dim

        nb_filter, nb_rows, nb_cols = self.filter_dim
        self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

        # self.b_h = K.zeros((nb_filter,))
        self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_x = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

        # hidden to hidden connections
        self.conv_h.build()
        # input to hidden connections
        self.conv_x.build()

        # self.max_pool = MaxPooling2D(pool_size=self.subsample, input_shape=hidden_dim)
        # self.max_pool.build()

        # self.trainable_weights = self.conv_h.trainable_weights + self.conv_x.trainable_weights
        self.params = self.conv_h.params + self.conv_x.params

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, X):
        batch_size = self._get_batch_size(X)
        hidden_dim = np.prod(self.output_dim)
        if K._BACKEND == 'theano':
            h = T.zeros((batch_size, hidden_dim))
            #h = K.zeros((batch_size, hidden_dim))
        else:
            h = K.zeros((batch_size, hidden_dim))
        return [h, ]

    def step(self, x, states):
        batch_size = self._get_batch_size(x)
        input_shape = (batch_size, ) + self.reshape_dim
        hidden_dim = (batch_size, ) + self.output_dim
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        Wx_t = self.conv_x(x_t, train=True)
        h_t = self.activation(Wx_t + self.conv_h(h_tm1, train=True))
        #h_t = K.batch_flatten(h_t)
        h_t = K.flatten(h_t)
        return h_t, [h_t, ]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "filter_dim": self.filter_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "return_sequences": self.return_sequences,
                  "reshape_dim": self.reshape_dim,
                  "go_backwards": self.go_backwards}
        base_config = super(ConvRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SubunitSlice(Layer):
    # This layer returns a given subunit response across time
    # Given a 5D tensor of the shape (samples, time, num_subunits, num_rows, num_cols),
    # this will return (samples, time, num_rows, num_cols) given a subunit index = 0, ..., num_subunits - 1
    input_ndim = 5
    def __init__(self, subunit_idx, **kwargs):
        self.subunit_idx = subunit_idx
        self.input = K.placeholder(ndim=5)
        super(SubunitSlice, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], input_shape[3], input_shape[4])

    def get_output(self, train=False):
        X = self.get_input(train)
        subunit_idx = self.subunit_idx
        finaloutput = K.subunit_slice(X, subunit_idx)
        return finaloutput

class TimeDistributedFlatten(Layer):
    # This layer reshapes input to be flat across timesteps (cannot be used as the first layer of a model)
    # Input shape: (num_samples, num_timesteps, *)
    # Output shape: (num_samples, num_timesteps, num_input_units)
    # Potential use case: For stacking after a Time Distributed Convolution/Max Pooling Layer or other Time Distributed Layer
    def __init__(self, **kwargs):
        super(TimeDistributedFlatten, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], np.prod(input_shape[2:]))

    def get_output(self, train=False):
        X = self.get_input(train)
        finaloutput = K.tdflatten(X)
        return finaloutput

class TimeDistributedConvolution2D(Layer):
    # This layer performs 2D Convolutions with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Deafault Output shape (Theano dim ordering): (num_samples, num_timesteps, num_filters, num_rows, num_cols), Note: num_rows and num_cols could have changed
    # Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer
    
    input_ndim = 5

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):
    
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=5)
        super(TimeDistributedConvolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[2]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[4]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        elif self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])
        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        input_dim = self.input_shape
        Y = K.collapsetime(X) #collapse num_samples and num_timesteps
        conv_out = K.conv2d(Y, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        finaloutput = K.expandtime(X, output)
        return finaloutput


    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(TimeDistributedConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _TimeDistributedPooling2D(Layer):
    '''Abstract class for different Time Distributed pooling 2D layers.
    '''
    input_ndim = 5

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(_TimeDistributedPooling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=5)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        elif self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], input_shape[2], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def get_output(self, train=False):
        X = self.get_input(train)
        input_dim = self.input_shape
        Y = K.collapsetime(X) #collapse num_samples and num_timesteps
        output = self._pooling_function(inputs=Y, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        finaloutput = K.expandtime(X, output)
        return finaloutput

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_TimeDistributedPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedMaxPooling2D(_TimeDistributedPooling2D):
    # This layer performs 2D Max Pooling with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Default Output shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer
    
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(TimeDistributedMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output

class TimeDistributedAveragePooling2D(_TimeDistributedPooling2D):
    # This layer performs 2D Average Pooling with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Default Output shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(TimeDistributedAveragePooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output
