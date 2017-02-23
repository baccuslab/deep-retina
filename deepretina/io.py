"""
Helper utilities for saving models and model outputs
"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv, path
from json import dumps
from collections import namedtuple
from itertools import product
from functools import wraps
from .utils import notify, allmetrics
from warnings import warn
import numpy as np
import inspect
import subprocess
import shutil
import time
import keras
import deepretina
import hashlib
import h5py

# Force matplotlib to not use any X-windows with the Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__all__ = ['Monitor', 'KerasMonitor', 'main_wrapper']

directories = {
    'dropbox': path.expanduser('~/Dropbox/deep-retina/saved/'),
    'database': path.expanduser('~/deep-retina-results/database'),
}


class Monitor:
    def __init__(self, name, model, experiment, readme, save_every):
        """Monitor base class

        Parameters
        ----------
        name : str
            A short string describing this model

        model : object
            A Keras or GLM model object

        experiment : experiments.Experiment
            A pointer to an Experiment class used to grab test data

        readme : str
            Saves this string as README.md

        save_every : int
            Parameters are saved only every save_every iterations
        """
        self.name = name
        self.model = model
        self.experiment = experiment
        self.save_every = save_every
        self.metrics = ('cc', 'lli', 'rmse', 'fev')

        # information about the machine this is running on
        machine = {
            'machine': uname()[1],
            'user': getenv('USER'),
            'timestamp': time.time(),
            'date': time.strftime("%Y-%m-%d"),
            'time': time.strftime("%H:%M:%S"),
            'keras': keras.__version__,
            'deep-retina': deepretina.__version__,
        }

        # generate a hash key for this model
        self.hashkey = md5('\n'.join(map(str, machine.values())))
        self.directory = ' '.join((self.hashkey, self.name))

        # keep track of the iteration with the best held out performance
        self.best = namedtuple('Best', ('iteration', 'lli'))(-1, -np.Inf)

        with notify('\nCreating directories and files for model {}'.format(self.hashkey)):

            # make the necessary folders on disk, for each item in directories
            for _, d in directories.items():
                mkdir(path.join(d, self.directory))

            # write some generic data to the file
            self._save_text('metadata.json', dumps(machine))
            self._save_text('experiment.json', dumps(self.experiment.info))
            self._save_text('README.md', readme)

            # start CSV files for train and validation performance
            headers = ','.join(('Epoch', 'Iteration') + tuple(map(str.upper, self.metrics))) + '\n'
            self._save_text('train.csv', headers)
            self._save_text('validation.csv', headers)

            # store results in a (new) h5 file
            with h5py.File(self._dbpath('results.h5'), 'x') as f:

                # store metadata
                f.attrs.update(machine)
                f.attrs['md5'] = self.hashkey

                # store experiment info
                f['cells'] = np.array(self.experiment.info['cells'])
                f['cells'].attrs.update(self.experiment.info)

                # initialize some datasets
                N = np.array(self.experiment.info['cells']).size
                f.create_dataset('iter', (0,), maxshape=(None,))
                f.create_dataset('epoch', (0,), maxshape=(None,))

                for k, m in product(('train', 'validation'), self.metrics):
                    f.create_dataset('/'.join((k, m)), (0, N), maxshape=(None, N))

                for fname, m in product(self.experiment._test_data.keys(), self.metrics):
                    f.create_dataset('/'.join(('test', fname, m)), (0, N), maxshape=(None, N))

    def _update_best(self, epoch, iteration):
        """Called when there is a new best iteration"""
        self.model.save_weights(self._dbpath('best_weights.h5'), overwrite=True)

    def cleanup(self, iteration, elapsed_time):
        """Called when the model has finished training"""
        print('Finished training model {} after {} iterations and {} hours.'
              .format(self.hashkey, iteration, elapsed_time / 3600.))

    def save(self, epoch, iteration, X_train, r_train, model_predict):
        """Saves relevant information for this epoch/iteration of training

        Saves the:
        - Model weights (as an hdf5 file)
        - Updated performance plots
        - Updated performance.csv file
        - Best performance and weights in a separate file

        Parameters
        ----------
        epoch : int
            Current epoch of training

        iteration : int
            Current iteration of training
        """
        #rhat_train = model_predict(X_train)
        #print(X_train.shape)
        rhat_train = model_predict({'stim': X_train}) #only updating this for graph model
        #print(rhat_train['loss'].shape)

        # training performance
        avg_train, all_train = allmetrics(r_train, model_predict({'stim': X_train}), self.metrics)
        data_row = [epoch, iteration] + [avg_train[metric] for metric in self.metrics]
        self._append_csv('train.csv', data_row)

        # validation performance
        (avg_val, all_val), r_val, rhat_val = self.experiment.validate(model_predict, self.metrics)
        data_row = [epoch, iteration] + [avg_val[metric] for metric in self.metrics]
        self._append_csv('validation.csv', data_row)

        # update the 'best' iteration we have seen, based on the validation log-likelihood
        if avg_val['lli'] > self.best.lli:
            self.best = namedtuple('Best', ('iteration', 'lli'))(iteration, avg_val['lli'])
            self._update_best(epoch, iteration)

        # evaluate test performance
        _, all_test = self.experiment.test(model_predict, self.metrics)

        # update h5 file
        self._save_h5(epoch, iteration, all_train, all_val, all_test)

        # plot the train / test firing rates
        cells = self.experiment.info['cells']
        if np.array(cells).size == 1:

            # for one cell
            filename = 'cell{}'.format(cells)
            plot_rates(iteration, self.experiment.dt,
                       train=(r_train, rhat_train['loss']),
                       validation=(r_val, rhat_val['loss']))
            self._save_figure(filename)

        else:
            # for all cells
            for ix, cell in enumerate(cells):
                filename = 'cell{}'.format(cell)
                plot_rates(iteration, self.experiment.dt,
                           train=(r_train[:, ix], rhat_train['loss'][:, ix]),
                           validation=(r_val[:, ix], rhat_val['loss'][:, ix]))
                self._save_figure(filename)

        # plot the performance curves
        for plottype in ('summary', 'traces'):
            filename = 'performance_{}'.format(plottype)
            with h5py.File(self._dbpath('results.h5'), 'r') as f:
                plot_performance(self.metrics, f, self.experiment.batches_per_epoch, plottype=plottype)
            self._save_figure(filename, dpi=100)

        # save the weights
        filename = 'epoch{:03d}_iter{:05d}_weights.h5'.format(epoch, iteration)
        self.model.save_weights(self._dbpath(filename))

    def _dbpath(self, filename):
        """Generates a full path to save the given file in the database directory"""
        return path.join(directories['database'], self.directory, filename)

    def _save_text(self, filename, text, dropbox=True):
        """Writes the given text to a file
        Optionally copies the file to Dropbox

        Parameters
        ----------
        filename : string
            Name of the file

        text : string
            Text to write to the file

        dropbox : boolean, optional
            Whether or not to copy the file to Dropbox (default: True)
        """
        # write the file, appending if it already exists
        with open(self._dbpath(filename), 'a') as f:
            f.write(text)

        # copy to dropbox
        if dropbox:
            self._copy_to_dropbox(filename)

    def _save_figure(self, filename, filetype='svg', dpi=100, dropbox=True):
        """Saves the current figure and copies it to Dropbox"""

        # set the file extension
        fname, ext = path.splitext(filename)
        filename = '.'.join((fname, filetype))

        # save the figure and close all
        plt.savefig(self._dbpath(filename),
                    format=filetype,
                    dpi=dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.close('all')

        # copy to dropbox
        if dropbox:
            self._copy_to_dropbox(filename)

    def _save_h5(self, epoch, iteration, all_train, all_val, all_test):
        """Updates the results.h5 file"""
        with h5py.File(self._dbpath('results.h5'), 'r+') as f:

            # helper function to extend a dataset along the first dimension
            def extend(key, value):
                shape = list(f[key].shape)
                shape[0] += 1
                f[key].resize(tuple(shape))
                f[key][-1] = value

            extend('epoch', epoch)
            extend('iter', iteration)

            for metric in self.metrics:
                extend('/'.join(('train', metric)), all_train[metric])
                extend('/'.join(('validation', metric)), all_val[metric])

                for fname, val in all_test.items():
                    extend('/'.join(('test', fname, metric)), all_test[fname][metric])

    def _append_csv(self, filename, row):
        """Appends the list of elements in row as a line in the CSV specified by filename"""
        self._save_text(filename, ','.join(map(str, row)) + '\n')

    def _copy_to_dropbox(self, filename):
        """Copy the given file to Dropbox. Overwrites existing destination files"""
        try:
            shutil.copy(self._dbpath(filename), path.join(directories['dropbox'], self.directory))
        except FileNotFoundError:
            warn('Could not copy {} to Dropbox.\n'.format(filename))


class KerasMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        """Builds a Monitor object to keep track of train/test performance

        Parameters
        ----------
        name : string
            a name for this model

        model : Keras model
            reference to the Keras model object

        experiment : Experiment
            a collection of experimental data (see experiments.py)

        readme : string
            a markdown formatted string to save as the README

        save_every : int
            how often to save (in terms of the number of batches)
        """
        super().__init__(*args, **kwargs)


def plot_rates(iteration, dt, **rates):
    """Plots the given pairs of firing rates"""

    # create the figure
    fig, axs = plt.subplots(len(rates), 1, figsize=(16, 10))

    # for now, manually choose indices to plot
    batchsize = rates['train'][0].shape[0]
    if batchsize > 3000:
        i0, i1 = (2000, 3000)
    else:
        i0, i1 = (0, batchsize - 1)
    inds = slice(i0, i1)

    for ax, key in zip(axs, sorted(rates.keys())):
        t = dt * np.arange(rates[key][0].size)
        ax.plot(t[inds], rates[key][0][inds], '-', color='powderblue', label='Data')
        ax.fill_between(t[inds], 0, rates[key][0][inds].ravel(), facecolor='powderblue', alpha=0.8)
        ax.plot(t[inds], rates[key][1][inds], '-', color='firebrick', label='Model')
        ax.set_title(str.upper(key) + ' [iter {}]'.format(iteration), fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=16)
        ax.set_xlim(t[i0], t[i1])
        despine(ax)

    plt.legend(loc='best', fancybox=True, frameon=True)
    plt.tight_layout()
    return fig


def plot_performance(metrics, results, batches_per_epoch, plottype='summary'):
    """Plots performance traces"""

    assert len(metrics) == 4, "plot_performance assumes there are four metrics to plot"

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    for metric, inds in zip(metrics, product((0, 1), repeat=2)):
        ax = axs[inds[0]][inds[1]]

        # the current epoch
        x = np.array(results['iter']).astype('float') / float(batches_per_epoch)
        for key, color, fmt in [('validation', 'lightcoral', '-'), ('train', 'skyblue', '--')]:
            res = np.array(results[key][metric])

            # plot the performance summary (mean + sem across cells)
            if plottype == 'summary':

                # only plot if we have some non-NaN values
                if not np.all(np.isnan(res)):
                    y = np.nanmean(res, axis=1)
                    ye = np.nanstd(res, axis=1) / np.sqrt(res.shape[1])
                    ax.fill_between(x, y - ye, y + ye, interpolate=True, alpha=0.2, color=color)
                    ax.plot(x, y, '-', color=color, label=key)

            # plot the performance traces (one curve for each cell)
            elif plottype == 'traces':
                ax.plot(x, res, fmt, alpha=0.5)

        # hard-coded y-scale for certain metrics
        if metric == 'fev':
            ax.set_ylim(-0.5, 0.5)

        ax.set_title(str.upper(metric), fontsize=20)
        ax.set_xlabel('Epoch', fontsize=16)
        despine(ax)

    if plottype == 'summary':
        axs[0][0].legend(loc='best', frameon=True, fancybox=True)
    plt.tight_layout()
    return fig


def main_wrapper(func):
    """Decorator for wrapping a main script

    Captures the source code in the script and generates a markdown-formatted README
    """
    @wraps(func)
    def mainscript(*args, **kwargs):

        # get information about this function call
        func.__name__
        source = inspect.getsource(func)
        commit = str(subprocess.check_output(["git", "describe", "--always"]), "utf-8")

        if 'description' in kwargs:
            description = kwargs.pop('description')
        else:
            description = input('Please enter a brief description of this model/experiment/script:\n')

        # build a markdown string containing this information
        readme = ['# deep-retina model training script',
                  '### description', description,
                  '### git commit', '[{}](https://github.com/baccuslab/deep-retina/commit/{})'.format(commit, commit),
                  '### function call', '```python\n{}(*{}, **{})\n```'.format(func.__name__, args, kwargs),
                  '### source', '```python', source, '```']
        kwargs['readme'] = '\n'.join(readme)

        func(*args, **kwargs)

    return mainscript


def despine(ax):
    """Gets rid of the top and right spines"""
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def md5(string, length=6):
    """Generates an md5 hash of the given string"""
    tmp = hashlib.md5()
    tmp.update(string.encode('ascii'))
    return tmp.hexdigest()[:length]
