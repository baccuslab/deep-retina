"""
Custom keras activations
"""
import numpy as np
from keras.engine import Layer
from keras import backend as K
from keras.initializers import Constant, Zeros

__all__ = ['ParametricSoftplus', 'psp', 'PSP', 'requ', 'ReQU', 'rbf', 'RBF', 'selu', 'SELU']


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
        self.alpha = self.add_weight(shape=input_shape[1:], initializer=Constant(self.alpha_init), name="alpha")
        self.beta = self.add_weight(shape=input_shape[1:], initializer=Constant(self.beta_init), name="beta")
        super().build(input_shape)

    def call(self, x):
        return K.softplus(self.beta * x) * self.alpha


def requ(x):
    """Rectified quadratic"""
    return K.square(K.relu(x))


def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


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
        self.theta = self.add_weight(shape=(len(self.params),), initializer=Zeros(), name="RBF_weights")
        super().build(input_shape)

    def call(self, x):
        A = K.stack([self.gaussian(x, *args) for args in self.params], axis=1)
        return K.softplus(K.dot(self.theta, A))


# aliases
psp = PSP = ParametricSoftplus
ReQU = requ
SELU = selu
rbf = RBF
