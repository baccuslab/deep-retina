"""
Helper utilities for saving models and model outputs

"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv, path
from json import dumps
from collections import defaultdict
from itertools import product
from . import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import shutil
import time
import theano
import keras
import deepretina
import hashlib
import h5py

__all__ = ['Monitor']

directories = {
    'dropbox': path.expanduser('~/Dropbox/deep-retina/saved/'),
    'database': path.expanduser('~/deep-retina-results/database'),
}


class Monitor:

    def __init__(self, name, model, data):

        # pointer to Keras model and experimental data
        self.name = name
        self.model = model
        self.data = data

        # store results in a dictionary
        self.results = {
            'iter': list(),
            'epoch': list(),
            'train': defaultdict(list),
            'test': defaultdict(list),
        }

        # metrics to use (names of functions in the metrics module)
        self.metrics = ('cc', 'lli', 'rmse', 'fev')

        # metadata related to this training instance
        self.metadata = {
            'machine': uname()[1],
            'user': getenv('USER'),
            'timestamp': time.time(),
            'date': time.strftime("%Y-%m-%d"),
            'time': time.strftime("%H.%M.%S"),
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

        # make the necessary folders on disk
        self.dirname = ' '.join((self.hashkey, self.name))
        for _, directory in directories.items():
            mkdir(path.join(directory, self.dirname))
        self.datadir = path.join(directories['database'], self.dirname)

        # write files to disk
        self.write('architecture.json', self.model.to_json())
        self.write('architecture.yaml', self.model.to_yaml())
        self.write('experiment.json', dumps(data.info))
        self.write('metadata.json', dumps(self.metadata))

        # start CSV files for performance
        headers = ','.join(('Epoch', 'Iteration') +
                           tuple(map(str.upper, self.metrics))) + '\n'
        self.write('train.csv', headers)
        self.write('test.csv', headers)

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
            f['test'] = 0
            f['iter'] = 0
            f['epoch'] = 0

    def savepath(self, filename):
        """Generates a fullpath to save the given file in the data directory"""
        return path.join(self.datadir, filename)

    def update_results(self):

        with h5py.File(self.savepath('results.h5'), 'r+') as f:

            del f['iter']
            f['iter'] = self.results['iter']

            del f['epoch']
            f['epoch'] = self.results['epoch']

            for dset in ('train', 'test'):

                # delete the dataset (TODO: perhaps handle this more elegantly)
                del f[dset]

                for metric in self.metrics:
                    f[dset + '/' + metric] = np.array(self.results[dset][metric])

    def save(self, epoch, iteration):
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
        self.results['iter'].append(iteration)
        self.results['epoch'].append(epoch)

        # compute the test metrics and predicted firing rates
        avg_scores, all_scores, r_train, rhat_train, r_test, rhat_test = self.test()

        for key in ('train', 'test'):

            # update performance CSV files with the average score across cells
            data_row = [epoch, iteration] + [avg_scores[key][metric] for metric in self.metrics]
            csv_row = ','.join(map(str, data_row)) + '\n'
            self.write(key + '.csv', csv_row)

            # append to results
            [self.results[key][metric].append(all_scores[key][metric])
             for metric in self.metrics]

        # save the weights
        filename = 'epoch{:03d}_iter{:05d}_weights.h5'.format(epoch, iteration)
        self.model.save_weights(self.savepath(filename))

        # update the results.h5 file
        self.update_results()

        # update the 'best' iteration we have seen
        if avg_scores['test']['cc'] > self.best[1]:

            # update the best iteration and held-out CC performance
            self.best = (iteration, avg_scores['test']['cc'])

            # save best weights
            self.model.save_weights(self.savepath('best_weights.h5'), overwrite=True)

        # plot the train / test firing rates
        for ix, cell in enumerate(self.data.info['cells']):
            filename = 'cell{}.jpg'.format(cell)
            plot_rates(self.data.dt,
                       train=(r_train[:, ix], rhat_train[:, ix]),
                       test=(r_test[:, ix], rhat_test[:, ix]))
            plt.savefig(self.savepath(filename), dpi=100, bbox_inches='tight')
            plt.close('all')
            self.copy_to_dropbox(filename)

        # plot the performance curves
        plot_performance(self.metrics, self.results)
        plt.savefig(self.savepath('performance.jpg'), dpi=100, bbox_inches='tight')
        plt.close('all')
        self.copy_to_dropbox('performance.jpg')

        # TODO: store results in SQL

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

    def test(self):
        """Evaluates metrics on the train and test datasets"""

        # performance on the entire holdout set
        r_test = self.data.test.y
        rhat_test = self.model.predict(self.data.test.X)

        # performance on a random subset of the training data
        training_sample_size = rhat_test.shape[0]
        inds = np.random.choice(self.data.train.y.shape[0],
                                training_sample_size, replace=False)
        r_train = self.data.train.y[inds]
        rhat_train = self.model.predict(self.data.train.X[inds, ...])

        # evalue using the given metrics (computes an average over the different cells)
        avg_scores = defaultdict(dict)
        all_scores = defaultdict(dict)
        for function in self.metrics:

            rates = {
                'train': (r_train.T, rhat_train.T),
                'test': (r_test.T, rhat_test.T),
            }

            # iterates over 'train' and 'test'
            for key, args in rates.items():

                # store the average across cells, and the individual scores for each cell
                avg, cells = getattr(metrics, function)(*args)
                avg_scores[key][function] = avg
                all_scores[key][function] = cells

        return avg_scores, all_scores, r_train, rhat_train, r_test, rhat_test


def plot_rates(dt, **rates):
    """Plots the given pairs of firing rates"""

    # create the figure
    fig, axs = plt.subplots(len(rates), 1)
    
    # for now, manually choose indices to plot
    i0, i1 = (2000, 4000)
    inds = slice(i0, i1)

    for ax, key in zip(axs, rates):
        t = dt * np.arange(rates[key][0].size)
        ax.plot(t[inds], rates[key][0][inds], '-', color='powderblue', label='Data')
        ax.plot(t[inds], rates[key][1][inds], '-', color='firebrick', label='Model')
        ax.set_title(str.upper(key), fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=16)
        ax.set_xlim(t[i0], t[i1])
        despine(ax)

    plt.legend(loc='best', fancybox=True, frameon=True)
    plt.tight_layout()
    return fig


def plot_performance(metrics, results):
    """Plots performance traces"""

    assert len(metrics) == 4, "plot_performance assumes there are four metrics to plot"

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    for metric, inds in zip(metrics, product((0, 1), repeat=2)):
        ax = axs[inds[0]][inds[1]]
        ax.plot(results['iter'], np.mean(results['test'][metric], axis=1), 'k-', label='test')
        ax.plot(results['iter'], np.mean(results['train'][metric], axis=1), 'k--', label='train')
        ax.set_title(str.upper(metric), fontsize=20)
        ax.set_xlabel('Iteration', fontsize=16)
        despine(ax)

    plt.legend(loc='best', frameon=True, fancybox=True)
    plt.tight_layout()
    return fig


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
