"""
Custom keras activations
"""
import numpy as np
from keras.engine import Layer
from keras import backend as K
from keras.initializers import Constant, Zeros


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
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=input_shape[1:], initializer=Constant(self.alpha_init))
        self.beta = self.add_weight(shape=input_shape[1:], initializer=Constant(self.beta_init))
        super().build(input_shape)

    def call(self, x):
        return K.softplus(self.beta * x) * self.alpha


class ReQU(Layer):
    def __init__(self, **kwargs):
        """Rectified quadratic nonlinearity

        Has the form: f(x) = [x]_+^2

        Parameters
        ----------
        alpha_init : array_like
            Initial values for the alphas (default: 0.2)

        beta_init : float
            Initial values for the betas (default: 5.0)
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return K.square(K.relu(x))


class RBF(Layer):
    def __init__(self, n, w, **kwargs):
        """Radial basis function (RBF) basis nonlinearity

        Parameters
        ----------
        n : int
            Number of basis functions to use

        w : float
            Width of the input range: basis functions span (-w, w)
        """
        xmin, xmax = (-w, w)
        dx = (xmax - xmin) / n
        centers = np.linspace(xmin, xmax, n)
        widths = np.linspace(-dx, dx, n) ** 2 + dx
        self.params = tuple(zip(centers, widths))
        super().__init__(**kwargs)

    @staticmethod
    def gaussian(x, mu, sigma):
        return K.exp(-(x - mu) ** 2 / sigma) / (2 * np.pi * sigma)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(len(self.params),), initializer=Zeros())
        super().build(input_shape)

    def call(self, x):
        A = K.stack([self.gaussian(x, *args) for args in self.params], axis=1)
        return K.softplus(K.dot(self.theta, A))
