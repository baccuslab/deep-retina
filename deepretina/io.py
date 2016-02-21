"""
Helper utilities for saving models and model outputs

"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv, path
from json import dumps
from collections import defaultdict
from itertools import product
from functools import wraps
from .utils import notify, allmetrics
from keras.utils import visualize_util
import numpy as np
import inspect
import subprocess
import shutil
import time
import theano
import keras
import deepretina
import hashlib
import h5py

# Force matplotlib to not use any X-windows with the Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__all__ = ['Monitor', 'main_wrapper']

directories = {
    'dropbox': path.expanduser('~/Dropbox/deep-retina/saved/'),
    'database': path.expanduser('~/deep-retina-results/database'),
}


class Monitor:
    def __init__(self, name, model, data, readme, save_every):
        """Builds a Monitor object to keep track of train/test performance

        Parameters
        ----------
        name : string
            a name for this model

        model : Keras model
            reference to the Keras model object

        data : Experiment
            a collection of experimental data (see experiments.py)

        readme : string
            a markdown formatted string to save as the README

        save_every : int
            how often to save (in terms of the number of batches)

        """

        # pointer to Keras model and experimental data
        self.name = name
        self.model = model
        self.data = data
        self.save_every = save_every

        # store results in a dictionary
        self.results = {
            'iter': list(),
            'epoch': list(),
            'train': defaultdict(list),
            'validation': defaultdict(list),
        }
        self.test_results = {fname: defaultdict(list) for fname in self.data._test_data.keys()}

        # metrics to use (names of functions in the metrics module)
        self.metrics = ('cc', 'lli', 'rmse', 'fev')

        # metadata related to this training instance
        self.metadata = {
            'machine': uname()[1],
            'user': getenv('USER'),
            'timestamp': time.time(),
            'date': time.strftime("%Y-%m-%d"),
            'time': time.strftime("%H:%M:%S"),
            'keras': keras.__version__,
            'deep-retina': deepretina.__version__,
            'theano': theano.__version__,
        }

        # generate a hash key for this model architecture
        hashstring = '\n'.join((self.metadata['date'],
                                self.metadata['time'],
                                self.metadata['machine'],
                                self.metadata['user'],
                                self.model.to_yaml()))
        self.hashkey = md5(hashstring)

        with notify('\nCreating directories and files for model {}'.format(self.hashkey)):

            # make the necessary folders on disk
            self.dirname = ' '.join((self.hashkey, self.name))
            for _, directory in directories.items():
                mkdir(path.join(directory, self.dirname))
            self.datadir = path.join(directories['database'], self.dirname)

            # writes files to disk (and copy them to dropbox)
            self.write('architecture.json', self.model.to_json())
            self.write('architecture.yaml', self.model.to_yaml())
            self.write('experiment.json', dumps(data.info))
            self.write('metadata.json', dumps(self.metadata))
            self.write('README.md', readme)

            # save model architecture as a figure and copy it to dropbox
            visualize_util.plot(self.model, to_file=self.savepath('architecture.png'))
            self.copy_to_dropbox('architecture.png')

            # start CSV files for performance
            headers = ','.join(('Epoch', 'Iteration') +
                               tuple(map(str.upper, self.metrics))) + '\n'
            self.write('train.csv', headers)
            self.write('validation.csv', headers)

        # keep track of the iteration with the best held out performance
        self.best = (-1, 0)

        # store results in a (new) h5 file
        with h5py.File(self.savepath('results.h5'), 'x') as f:

            # store metadata
            f.attrs.update(self.metadata)
            f.attrs['md5'] = self.hashkey

            # store experiment info
            f['cells'] = np.array(data.info['cells'])
            f['cells'].attrs.update(data.info)

            # initialize some datasets
            f['train'] = 0
            f['validation'] = 0
            f['iter'] = 0
            f['epoch'] = 0

            # store the test performance for each filetype
            for fname in self.test_results.keys():
                f['test/' + fname] = 0

    def savepath(self, filename):
        """Generates a fullpath to save the given file in the data directory"""
        return path.join(self.datadir, filename)

    def update_results(self):

        with h5py.File(self.savepath('results.h5'), 'r+') as f:

            del f['iter']
            f['iter'] = self.results['iter']

            del f['epoch']
            f['epoch'] = self.results['epoch']

            for dset in ('train', 'validation'):

                # delete the dataset (TODO: perhaps handle this more elegantly)
                del f[dset]

                for metric in self.metrics:
                    f[dset + '/' + metric] = np.array(self.results[dset][metric])

            del f['test']
            for key, result in self.test_results.items():
                for metric in self.metrics:
                    f['/'.join(('test', key, metric))] = np.array(result[metric])

    def save(self, epoch, iteration, r_train, rhat_train):
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

        r : array_like
            The true responses on this training batch

        rhat : array_like
            The model responses on this training batch
        """
        self.results['iter'].append(iteration)
        self.results['epoch'].append(epoch)

        # STORE TRAINING PERFORMANCE
        avg_train, all_train = allmetrics(r_train, rhat_train, self.metrics)
        data_row = [epoch, iteration] + [avg_train[metric] for metric in self.metrics]
        self.write('train.csv', ','.join(map(str, data_row)) + '\n')
        [self.results['train'][metric].append(all_train[metric]) for metric in self.metrics]

        # STORE VALIDATION PERFORMANCE
        (avg_val, all_val), r_val, rhat_val = self.data.validate(self.model.predict, self.metrics)
        data_row = [epoch, iteration] + [avg_val[metric] for metric in self.metrics]
        self.write('validation.csv', ','.join(map(str, data_row)) + '\n')
        [self.results['validation'][metric].append(all_val[metric]) for metric in self.metrics]

        # save the weights
        filename = 'epoch{:03d}_iter{:05d}_weights.h5'.format(epoch, iteration)
        self.model.save_weights(self.savepath(filename))

        # EVALUATE TEST PERFORMANCE
        avg_test, all_test = self.data.test(self.model.predict, self.metrics)
        [self.test_results[key][metric].append(all_test[key][metric])
         for metric in self.metrics
         for key in self.test_results]

        # update the results.h5 file
        self.update_results()

        # update the 'best' iteration we have seen
        if avg_val['cc'] > self.best[1]:

            # update the best iteration and held-out CC performance
            self.best = (iteration, avg_val['cc'])

            # save best weights
            self.model.save_weights(self.savepath('best_weights.h5'), overwrite=True)

        # plot the train / test firing rates
        for ix, cell in enumerate(self.data.info['cells']):
            filename = 'cell{}'.format(cell)
            plot_rates(iteration, self.data.dt,
                       train=(r_train[:, ix], rhat_train[:, ix]),
                       validation=(r_val[:, ix], rhat_val[:, ix]))
            self._save_and_copy(filename, filetype='jpg', dpi=100)

        # plot the performance curves
        for plottype in ('summary', 'traces'):
            filename = 'performance_{}'.format(plottype)
            plot_performance(self.metrics, self.results, self.data.batchsize, plottype=plottype)
            self._save_and_copy(filename, filetype='jpg', dpi=100)

    def write(self, filename, text, copy=True):
        """Writes the given text to a file
        Optionally copies the file to Dropbox

        Parameters
        ----------
        filename : string
            Name of the file

        text : string
            Text to write to the file

        copy : boolean, optional
            Whether or not to copy the file to Dropbox (default: True)
        """
        # write the file, appending if it already exists
        with open(self.savepath(filename), 'a') as f:
            f.write(text)

        # copy to dropbox
        if copy:
            self.copy_to_dropbox(filename)

    def copy_to_dropbox(self, filename):
        """Copy the given file to Dropbox. Overwrites existing destination files"""
        try:
            shutil.copy(self.savepath(filename), path.join(directories['dropbox'], self.dirname))
        except FileNotFoundError:
            print('\n*******\nWarning\n*******')
            print('Could not copy {} to Dropbox.\n'.format(filename))

    def _save_and_copy(self, filename, filetype='jpg', dpi=100):
        """Saves the current figure as a jpg and copies it to Dropbox"""
        filename = filename + '.' + filetype
        plt.savefig(self.savepath(filename), dpi=dpi, bbox_inches='tight')
        plt.close('all')
        self.copy_to_dropbox(filename)


def plot_rates(iteration, dt, **rates):
    """Plots the given pairs of firing rates"""

    # create the figure
    fig, axs = plt.subplots(len(rates), 1, figsize=(16, 10))

    # for now, manually choose indices to plot
    batchsize = rates['train'][0].shape[0]
    if batchsize > 3000:
        i0, i1 = (2000, 3000)
    else:
        i0, i1 =(0, batchsize-1)
    inds = slice(i0, i1)

    for ax, key in zip(axs, sorted(rates.keys())):
        t = dt * np.arange(rates[key][0].size)
        ax.plot(t[inds], rates[key][0][inds], '-', color='powderblue', label='Data')
        ax.fill_between(t[inds], 0, rates[key][0][inds], facecolor='powderblue', alpha=0.8)
        ax.plot(t[inds], rates[key][1][inds], '-', color='firebrick', label='Model')
        ax.set_title(str.upper(key) + ' [iter {}]'.format(iteration), fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=16)
        ax.set_xlim(t[i0], t[i1])
        despine(ax)

    plt.legend(loc='best', fancybox=True, frameon=True)
    plt.tight_layout()
    return fig


def plot_performance(metrics, results, batchsize, plottype='summary'):
    """Plots performance traces"""

    assert len(metrics) == 4, "plot_performance assumes there are four metrics to plot"

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    for metric, inds in zip(metrics, product((0, 1), repeat=2)):
        ax = axs[inds[0]][inds[1]]

        # the current epoch
        x = np.array(results['iter']) / float(batchsize)
        for key, color, fmt in [('validation', 'lightcoral', '-'), ('train', 'skyblue', '--')]:
            res = np.array(results[key][metric])

            # plot the performance summary (mean + sem across cells)
            if plottype == 'summary':
                y = np.nanmean(res, axis=1)
                ye = np.nanstd(res, axis=1) / np.sqrt(res.shape[1])
                ax.fill_between(x, y - ye, y + ye, interpolate=True, alpha=0.2, color=color)
                ax.plot(x, y, '-', color=color, label=key)

            # plot the performance traces (one curve for each cell)
            elif plottype == 'traces':
                ax.plot(x, res, fmt, alpha=0.5)

        ax.set_title(str.upper(metric), fontsize=20)
        ax.set_xlabel('Epoch', fontsize=16)
        despine(ax)

    if plottype == 'summary':
        axs[0][0].legend(loc='best', frameon=True, fancybox=True)
    plt.tight_layout()
    return fig


def main_wrapper(func):
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
        kwargs['readme'] = '\n'.join(['# deep-retina model training script',
                                      '### description', description,
                                      '### git commit', '[{}](https://github.com/baccuslab/deep-retina/commit/{})'.format(commit, commit),
                                      '### function call', '```python\n{}(*{}, **{})\n```'.format(func.__name__, args, kwargs),
                                      '### source', '```python', source, '```',
                                      ])

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
