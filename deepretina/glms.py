"""
Generalized linear model

Implementation of GLMs with post-spike history and coupling filters
(see Pillow et. al. 2008 for details)
"""
import numpy as np
from descent.utils import destruct, restruct
from descent.algorithms import RMSProp

__all__ = ['GLM']


class GLM:
    def __init__(self, shape, lr=1e-4):
        """GLM model class"""

        # initialize parameters
        self.theta_init = {
            'filter': np.random.randn(*shape) * 1e-6,
            'bias': 0.0,
        }

        # initialize optimizer
        self.opt = RMSProp(destruct(self.theta_init).copy(), lr=lr)

    @property
    def theta(self):
        return restruct(self.opt.xk, self.theta_init)

    def predict(self, X):
        """Predicts the firing rate given the stimulus"""
        return np.exp(self.project(X))

    def train_on_batch(self, X, y):
        """Updates the parameters on the given batch"""

        # compute the objective and gradient
        obj, gradient = self.loss(X, y)

        # pass the gradient to the optimizer
        self.opt(destruct(gradient))

        return obj, gradient

    def loss(self, X, y):
        """Gets the objective and gradient for the given batch of data"""
        u = self.project(X)
        rhat = np.exp(u)

        # compute the objective
        obj = (rhat - y * u).mean()

        # compute gradient
        factor = rhat - y
        gradient = {}
        gradient['bias'] = factor.mean()
        gradient['filter'] = np.tensordot(factor, X, axes=1) / float(X.shape[0])

        return obj, gradient

    def project(self, X):
        """Projects the given stimulus onto the filter parameters"""
        return np.tensordot(X, self.theta['filter'], axes=self.theta['filter'].ndim) + self.theta['bias']
