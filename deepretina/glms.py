"""
Generalized linear model

Implementation of GLMs with post-spike history and coupling filters
(see Pillow et. al. 2008 for details)
"""
import numpy as np
import h5py
from os import path
from descent.utils import destruct, restruct
from descent import rmsprop

__all__ = ['GLM']


class GLM:
    def __init__(self, filter_shape, coupling_history, ncells, lr=1e-4, l2=0.0, dt=1e-2):
        """GLM model class

        Parameters
        ----------
        filter_shape : tuple
            The dimensions of the stimulus filter, e.g. (nt, nx, ny)

        coupling_history : int
            How many timesteps to include in the coupling filter

        lr : float, optional
            Learning rate for RMSprop (Default: 1e-4)

        l2 : float, optional
            l2 regularization penalty on the weights (Default: 0.0)
        """
        self.dt = dt

        # initialize parameters
        self.theta_init = {
            'filter': np.random.randn(*(filter_shape + (ncells,))) * 1e-6,
            'bias': np.zeros(ncells),
            'history': np.random.randn(coupling_history, ncells, ncells) * 1e-6
        }

        # initialize optimizer
        self.opt = rmsprop(destruct(self.theta_init).copy(), lr=lr)

        # add regularization
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
        """Gets the dictionary of parameters"""
        return restruct(self.opt.xk, self.theta_init)

    def set_theta(self, theta):
        """Manually sets the values for the parameters"""
        self.opt.xk = destruct(theta).copy()

    def generator(self, X):
        """Gets the generator signal (pre-nonlinearity)"""
        nsamples = X.shape[0]
        nhistory = self.theta['history'].shape[0]

        # project the stimulus onto the stimulus filter
        nax = self.theta['filter'].ndim - 1
        u = np.tensordot(X, self.theta['filter'], axes=nax) + self.theta['bias']
        spikes = np.empty_like(u)

        # store the augmented history matrix
        H = np.zeros((X.shape[0],) + self.theta['history'].shape[:-1])

        # incrementally apply the spike history and coupling filters
        for t in range(nsamples):

            # pad the spikes
            if t < nhistory:
                spikepad = np.pad(spikes[:t], ((nhistory - t, 0), (0, 0)), 'constant')
            else:
                spikepad = spikes[(t - nhistory):t]
            H[t] = spikepad

            # project spike history onto coupling filters
            u[t] += np.tensordot(H[t], self.theta['history'], axes=2)

            # draw poisson spikes for this time point
            spikes[t] = np.random.poisson(self.dt * texp(u[t]))

        return u, H

    def predict(self, X):
        """Predicts the firing rate given a stimulus"""
        return texp(self.generator(X)[0])

    def train_on_batch(self, X, y):
        """Updates the parameters on the given batch

        (with the corresponding regularization penalties)
        """
        # compute the objective and gradient
        objective, gradient = self.loss(X, y)

        # update objective and gradient with the l2 penalty
        for key in gradient.keys():
            objective += 0.5 * self.l2[key] * np.linalg.norm(self.theta[key].ravel(), 2) ** 2
            gradient[key] += self.l2[key] * self.theta[key]

        # pass the gradient to the optimizer
        self.opt(destruct(gradient))

        return objective, gradient

    def loss(self, X, y):
        """Gets the objective and gradient for the given batch of data

        (ignores the l2 regularization penalty)
        """
        # forward pass
        u, H = self.generator(X)
        yhat = texp(u)

        # compute the objective
        objective = (yhat - y * u).mean()

        # compute gradient
        factor = yhat - y
        T = float(factor.size)
        gradient = {
            'bias': factor.mean(axis=0) / float(self.theta['bias'].size),
            'filter': np.tensordot(X, factor, axes=(0, 0)) / T,
            'history': np.tensordot(H, factor, axes=(0, 0)) / T,
        }

        return objective, gradient

    def get_f_df(self, X, y, regularize=True):
        """returns an f_df function (for use with check_grad, for example)"""
        def f_df(theta):
            self.set_theta(theta)
            objective, grad = self.loss(X, y)
            if regularize:
                for key in grad.keys():
                    objective += 0.5 * self.l2[key] * np.linalg.norm(self.theta[key].ravel(), 2) ** 2
                    grad[key] += self.l2[key] * self.theta[key]
            return objective, grad
        return f_df

    def save_weights(self, filepath, overwrite=False):
        """Saves weights to an HDF5 file"""

        if not overwrite and path.isfile(filepath):
            raise FileExistsError("The file '{}' already exists\n(did you mean to set overwrite=True ?)".format(filepath))

        with h5py.File(filepath, 'w') as f:
            for key, value in self.theta.items():
                dset = f.create_dataset(key, value.shape, dtype=value.dtype)
                dset[:] = value


def test_glm():
    from itertools import product
    from tqdm import trange

    # parameters
    nt = 1          # time points in the stimulus filter
    nx = 3          # filter spatial dimension
    nc = 2          # number of cells
    nh = 20         # number of time points in the history (coupling) filter

    # generate a 'true' model
    theta_star = {}
    theta_star['filter'] = np.random.randn(nt, nx, nx, nc)
    theta_star['filter'] /= np.linalg.norm(theta_star['filter'].ravel())
    theta_star['history'] = np.zeros((nh, nc, nc))
    for i, j in product(range(nc), range(nc)):
        theta_star['history'][:, i, j] = 0.1 * np.sin(np.linspace(0, 2 * np.pi, nh) + 2 * np.pi * np.random.rand())
    theta_star['bias'] = np.random.rand(nc) - 2.0
    true_model = GLM((nt, nx, nx), nh, nc)
    true_model.set_theta(theta_star)

    # generate data from the true model
    def datagen(niter=200, nsamples=10000):
        for _ in trange(int(niter)):
            X = np.random.randn(nsamples, nt, nx, nx)
            y = true_model.predict(X)
            yield (X, y)

    X, rstar = next(datagen())
    print('Mean firing rates: {}'.format(rstar.mean(axis=0)))
    print('Max firing rates: {}'.format(rstar.max(axis=0)))

    # fit a model to data from the true model
    model = GLM((nt, nx, nx), nh, nc, lr=1e-3)
    objs = list()
    for X, y in datagen():
        fobj = model.train_on_batch(X, y)[0]
        objs.append(fobj)

    return true_model, model, np.array(objs)


def texp(x, vmin=-20, vmax=20):
    """Truncated exponential"""
    return np.exp(x.clip(vmin, vmax))

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # fit a GLM to simulated data
    true_model, model, fobj = test_glm()

    plt.plot(fobj)
