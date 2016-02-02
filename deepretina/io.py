"""
Helper utilities for saving models and model outputs

"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv
from os.path import join, expanduser
from json import dumps
from collections import defaultdict
from . import metrics
import numpy as np
import shutil
import time
import theano
import keras
import deepretina
import hashlib

__all__ = ['Monitor']

directories = {
    'dropbox': expanduser('~/Dropbox/deep-retina/saved/'),
    'database': expanduser('~/deep-retina-results/database'),
}


class Monitor:

    def __init__(self, name, model, data):

        # pointer to Keras model and experimental data
        self.name = name
        self.model = model
        self.data = data

        # storage for keeping track of model metrics
        self.results = defaultdict(list)
        self.best = (-1, 0)

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

        # build a hash for this model architecture
        tmp = hashlib.md5()
        tmp.update('\n'.join((self.start['date'],
                              self.start['time'],
                              self.model.to_yaml()).encode('ascii')))
        self.hashkey = tmp.hexdigest()[:6]

        # make the necessary folders
        self.datadir = ' '.join((self.hashkey, self.name))
        for _, directory in directories.items():
            mkdir(join(directory, self.datadir))

        # write files to disk
        self.write('architecture.json', self.model.to_json())
        self.write('architecture.yaml', self.model.to_yaml())
        self.write('experiment.json', dumps(data.info))
        self.write('metadata.json', dumps(self.metadata))

    def save(self, epoch, iteration):

        # compute the test metrics and predicted firing rates
        scores, r_train, rhat_train, r_test, rhat_test = self.test()

        # store
        self.results['iter'].append(iteration)
        self.results['epoch'].append(epoch)
        for key in ('train', 'test'):
            for metric in ('cc', 'lli', 'fev', 'rmse'):
                self.results[':'.join(key, metric)].append(scores[key][metric])

        # save the weights
        filename = 'epoch{:03d}_iter{:05d}_weights.h5'.format(epoch, iteration)
        self.model.save_weights(filename)

        # update the 'best' iteration we have seen
        if scores['test']['cc'] > self.best[1]:
            self.best = (iteration, scores['test']['cc'])

            # TODO: save best weights

        # TODO: plot the train / test firing rates and save in a figure
        # TODO: plot the performance over time
        # TODO: store results in SQL
        # TODO: store results locally in an hdf5 file

    def write(self, filename, text, copy=True):
        """Writes the given text to a file"""
        fullpath = join(directories['database'], self.datadir, filename)

        with open(fullpath, 'w') as f:
            f.write(text)

        # copy to dropbox
        if copy:
            shutil.copy(fullpath, directories['dropbox'])

    @property
    def rhat(self):
        return self.model.predict(self.data.test.X)

    def test(self):

        # performance on the entire holdout set
        r_test = self.data.test.y
        rhat_test = self.model.predict(self.data.test.X)

        # performance on a random subset of the training data
        training_sample_size = rhat_test.shape[0]
        inds = np.random.choice(self.training.y.shape[0],
                                training_sample_size, replace=False)
        r_train = self.data.train.y[inds]
        rhat_train = self.model.predict(self.data.train.X[inds, ...])

        # evalue using the given metrics (computes an average over the different cells)
        scores = {}
        for function in ('cc', 'lli', 'fev', 'rmse'):
            scores['train'][function] = getattr(metrics, function)(r_train, rhat_train)
            scores['test'][function] = getattr(metrics, function)(r_test, rhat_test)

        return scores, r_train, rhat_train, r_test, rhat_test
