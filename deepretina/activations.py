from __future__ import absolute_import

import numpy as np
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras.legacy import interfaces

class ParametricSoftplus(Layer):
    '''Parametric Softplus of the form: alpha * log(1 + exp(beta * X))
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        beta_init: float. Initial values of the beta weights.
        weights: initial weights, as a list of 2 numpy arrays.
    # References:
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
    '''
    def __init__(self, alpha_init=0.2, 
                 beta_init=5.0,
                 weights=None, 
                 shared_axes=None,
                 **kwargs):
        self.alpha = K.cast_to_floatx(alpha_init)
        self.beta = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        super(ParametricSoftplus, self).__init__(**kwargs)
        self.supports_masking = True

        #self.alpha_init = initializers.get(alpha_init)
        #self.beta_init = initializers.get(beta_init)
        #self.initial_weights = initializers.get(weights)

    def call(self, inputs):
        return K.softplus(self.beta * inputs) * self.alpha


    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha_init': self.alpha,
                  'beta_init': self.beta}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
