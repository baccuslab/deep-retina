"""
Generalized linear model

Implementation of GLMs with post-spike history and coupling filters
(see Pillow et. al. 2008 for details)
"""
import numpy as np
import h5py
from os import path
from descent.utils import destruct, restruct
from descent.algorithms import RMSProp

__all__ = ['GLM']


class GLM:
    def __init__(self, shape, lr=1e-4):
        """GLM model class"""

        # initialize parameters
        self.theta_init = {
            'filter': np.random.randn(*shape) * 1e-6,
            'bias': np.array([0.0]),
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

    def save_weights(self, filepath, overwrite=False):
        """Saves weights to an HDF5 file"""

        if not overwrite and path.isfile(filepath):
            raise FileExistsError("The file '{}' already exists\n(did you mean to set overwrite=True ?)".format(filepath))

        with h5py.File(filepath, 'w') as f:
            for key, value in self.theta.items():
                dset = f.create_dataset(key, value.shape, dtype=value.dtype)
                dset[:] = value
