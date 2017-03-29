"""
Custom keras activations
"""
from keras.engine import Layer
from keras import backend as K


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

    def call(self, inputs):
        return K.softplus(self.beta * inputs) * self.alpha

    def get_config(self):
        config = {'alpha': float(self.alpha), 'beta': float(self.beta)}
        base_config = super().get_config()
        return base_config.update(config)
