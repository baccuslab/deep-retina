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
    def __init__(self, shape, lr=1e-4, l2=0.0):
        """GLM model class

        Parameters
        ----------
        shape : tuple
            The dimensions of the stimulus filter, e.g. (nt, nx, ny)

        lr : float, optional
            Learning rate for RMSprop (Default: 1e-4)

        l2 : float, optional
            l2 regularization penalty on the weights (Default: 0.0)
        """

        # initialize parameters
        self.theta_init = {
            'filter': np.random.randn(*shape) * 1e-6,
            'bias': np.array([0.0]),
        }

        # initialize optimizer
        self.opt = RMSProp(destruct(self.theta_init).copy(), lr=lr)

        # regularization
        if type(l2) is float:
            # same value for all keys
            self.l2 = {key: l2 for key in self.theta_init.keys()}

        elif type(l2) is dict:

            # default value is zero for every parameter
            self.l2 = {key: 0.0 for key in self.theta_init.keys()}

            # update with the given values
            self.l2.update(l2)

        else:
            raise ValueError("l2 keyword argument must be a float or a dictionary")

    @property
    def theta(self):
        return restruct(self.opt.xk, self.theta_init)

    def set_theta(self, theta):
        self.opt.xk = destruct(theta).copy()

    def predict(self, X):
        """Predicts the firing rate given the stimulus"""
        return np.exp(self.project(X))

    def train_on_batch(self, X, y):
        """Updates the parameters on the given batch

        (with the corresponding regularization penalties)
        """

        # compute the objective and gradient
        obj, gradient = self.loss(X, y)

        # update objective and gradient with the l2 penalty
        for key in gradient.keys():
            obj += 0.5 * self.l2[key] * np.linalg.norm(self.theta[key].ravel(), 2) ** 2
            gradient[key] += self.l2[key] * self.theta[key]

        # pass the gradient to the optimizer
        self.opt(destruct(gradient))

        return obj, gradient

    def loss(self, X, y):
        """Gets the objective and gradient for the given batch of data

        (ignores the l2 regularization penalty)
        """
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
