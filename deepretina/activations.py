"""
Custom keras activations
"""
from keras.engine import Layer
from keras import backend as K
from keras.initializers import Constant


class ParametricSoftplus(Layer):
    def __init__(self, alpha_init=0.2, beta_init=5., **kwargs):
        """Parametric softplus nonlinearity

        Has the form: f(x) = alpha * log(1 + exp(beta * x))

        Parameters
        ----------
        alpha_init : array_like
            Initial values for the alphas (default: 0.2)

        beta_init : float
            Initial values for the betas (default: 5.0)
        """
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.beta_init = beta_init

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=input_shape[1:], initializer=Constant(self.alpha_init))
        self.beta = self.add_weight(shape=input_shape[1:], initializer=Constant(self.beta_init))
        super().build(input_shape)

    def call(self, inputs):
        return K.softplus(self.beta * inputs) * self.alpha

    def get_config(self):
        #config = {'alpha': self.alpha, 'beta': self.beta}
        config = {}
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))
